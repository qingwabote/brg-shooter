using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public class MaterialOverrideBRG : MonoBehaviour
{
    public Material Material;
    public Mesh Mesh;

    public int TotalInstances = 1024;

    private int m_WindowInstances;
    private BatchID[] m_BatchIDs;
    private BatchMaterialID m_MaterialID;
    private BatchMeshID m_MeshID;

    void Start()
    {
        var instanceSize = (3 /* mat4x3 */ + 1 /* color */) * 16 /* vec4 */;
        var totalSize = instanceSize * TotalInstances;
        var windowSize = BatchRendererGroup.GetConstantBufferMaxWindowSize();
        var windowInstances = windowSize / instanceSize;
        Debug.Log($"windowSize: {windowSize} windowInstances: {windowInstances}");

        var metadata = new NativeArray<MetadataValue>(2, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        metadata[0] = new MetadataValue
        {
            NameID = Shader.PropertyToID("unity_ObjectToWorld"),
            Value = 0u | 0x80000000
        };
        metadata[1] = new MetadataValue
        {
            NameID = Shader.PropertyToID("_BaseColor"),
            Value = (3u * 16u * (uint)windowInstances) | 0x80000000
        };

        var brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
        brg.SetGlobalBounds(new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f)));

        var identity = float3x3.identity;
        var vecArray = new NativeArray<float4>(totalSize / 16/* vec4 */, Allocator.Temp, NativeArrayOptions.ClearMemory);
        var windows = TotalInstances / windowInstances;
        var left = TotalInstances % windowInstances;
        m_BatchIDs = new BatchID[windows + (left > 0 ? 1 : 0)];
        var buffer = new GraphicsBuffer(BatchRendererGroup.BufferTarget == BatchBufferTarget.ConstantBuffer ? GraphicsBuffer.Target.Constant : GraphicsBuffer.Target.Raw, totalSize / 16, 16);
        var random = new Unity.Mathematics.Random(2);
        for (int w = 0; w <= windows; w++)
        {
            var offset = windowSize / 16/* vec4 */ * w;
            var instances = w == windows ? left : windowInstances;
            for (int i = 0; i < instances; i++)
            {
                var pos = new float3(random.NextFloat(-3, 3), random.NextFloat(-3, 3), random.NextFloat(-3, 3));

                vecArray[i * 3 + 0 + offset] = new float4(identity.c0.x, identity.c0.y, identity.c0.z, identity.c1.x);
                vecArray[i * 3 + 1 + offset] = new float4(identity.c1.y, identity.c1.z, identity.c2.x, identity.c2.y);
                vecArray[i * 3 + 2 + offset] = new float4(identity.c2.z, pos.x, pos.y, pos.z);

                vecArray[3 * instances + i + offset] = new float4(random.NextFloat(0, 1), random.NextFloat(0, 1), random.NextFloat(0, 1), 1);
            }

            m_BatchIDs[w] = brg.AddBatch(metadata, buffer.bufferHandle, (uint)(windowSize * w), BatchRendererGroup.BufferTarget == BatchBufferTarget.ConstantBuffer ? (uint)windowSize : 0);
        }


        buffer.SetData(vecArray);

        m_MaterialID = brg.RegisterMaterial(Material);
        m_MeshID = brg.RegisterMesh(Mesh);


        m_WindowInstances = windowInstances;
    }

    unsafe public JobHandle OnPerformCulling(BatchRendererGroup rendererGroup, BatchCullingContext cullingContext, BatchCullingOutput cullingOutput, IntPtr userContext)
    {
        var windows = TotalInstances / m_WindowInstances;
        var left = TotalInstances % m_WindowInstances;

        var outputDraws = (BatchCullingOutputDrawCommands*)cullingOutput.drawCommands.GetUnsafePtr();

        var ranges = (BatchDrawRange*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>() * 1, UnsafeUtility.AlignOf<BatchDrawRange>(), Allocator.TempJob);
        ranges[0] = new BatchDrawRange
        {
            drawCommandsBegin = 0,
            drawCommandsCount = (uint)(windows + (left > 0 ? 1 : 0)),
            filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffff }
        };
        outputDraws->drawRanges = ranges;
        outputDraws->drawRangeCount = 1;

        var visibleInstances = (int*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<int>() * m_WindowInstances, UnsafeUtility.AlignOf<int>(), Allocator.TempJob);
        for (int i = 0; i < m_WindowInstances; i++)
        {
            visibleInstances[i] = i;
        }
        outputDraws->visibleInstances = visibleInstances;
        outputDraws->visibleInstanceCount = m_WindowInstances;

        var batchDraws = (BatchDrawCommand*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>() * windows + (left > 0 ? 1 : 0), UnsafeUtility.AlignOf<BatchDrawCommand>(), Allocator.TempJob);
        for (int w = 0; w <= windows; w++)
        {
            var instances = w == windows ? left : m_WindowInstances;
            if (instances == 0)
            {
                break;
            }

            batchDraws[w] = new BatchDrawCommand
            {
                visibleOffset = 0,    // all draw command is using the same {0,1,2,3...} visibility int array
                visibleCount = (uint)instances,
                batchID = m_BatchIDs[w],
                materialID = m_MaterialID,
                meshID = m_MeshID,
                submeshIndex = 0,
                splitVisibilityMask = 0xff,
                flags = BatchDrawCommandFlags.None,
                sortingPosition = 0
            };
        }

        outputDraws->drawCommands = batchDraws;
        outputDraws->drawCommandCount = windows + (left > 0 ? 1 : 0);

        outputDraws->instanceSortingPositions = null;
        outputDraws->instanceSortingPositionFloatCount = 0;

        return new JobHandle();
    }
}
