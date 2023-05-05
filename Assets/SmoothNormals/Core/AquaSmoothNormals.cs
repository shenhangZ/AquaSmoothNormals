using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;

namespace AquaSys.SmoothNormals
{
    public enum SmoothedNormalType
    {
        Normal,
        Compressed
    }

    public class AquaSmoothNormals
    {
        public static Vector2[] ComputeSmoothedNormals(Mesh mesh)
        {
            Vector3[] vertices = mesh.vertices;
            Vector3[] normals = mesh.normals;
            Vector4[] tans = mesh.tangents;

            int vertexCount = vertices.Length;

            if (tans.Length == 0)
            {
                Debug.LogError($"{mesh.name} don't have tangents.");
                return null;
            }

            // CalcSmoothedNormals
            // 它是一个高性能、线程安全的 hashmap 数据结构，特别适用于多线程并行计算场景。
            UnsafeParallelHashMap<Vector3, Vector3> smoothedNormalsMap = new UnsafeParallelHashMap<Vector3, Vector3>(vertexCount, Allocator.Persistent);
            for (int i = 0; i < vertexCount; i++)
            {
                // 重复存在则继续叠加法线值
                if (smoothedNormalsMap.ContainsKey(vertices[i]))
                {
                    smoothedNormalsMap[vertices[i]] = smoothedNormalsMap[vertices[i]] + normals[i];
                }
                else
                {
                    smoothedNormalsMap.Add(vertices[i], normals[i]);
                }
            }

            //BakeSmoothedNormals
            //`Allocator.Persistent` 的作用是在整个程序执行期间持久分配内存。换句话说，当使用 `Allocator.Persistent` 分配内存时
            //分配的内存将不会像其他分配策略那样自动释放。这种策略通常用于分配需要在整个程序执行期间一直存在的资源。
            NativeArray<Vector3> normalsNative = new NativeArray<Vector3>(normals, Allocator.Persistent);
            NativeArray<Vector3> vertrxNative = new NativeArray<Vector3>(vertices, Allocator.Persistent);
            NativeArray<Vector4> tangents = new NativeArray<Vector4>(tans, Allocator.Persistent);
            NativeArray<Vector2> bakedNormals = new NativeArray<Vector2>(vertexCount, Allocator.Persistent);

            BakeNormalJob bakeNormalJob = new BakeNormalJob(vertrxNative, normalsNative, tangents, smoothedNormalsMap, bakedNormals);
            // Schedule() 方法用于将要执行的任务添加到 Job 系统
            // vertexCount ：表示将任务分成多少个批次，其中每个批次包含一个顶点。这意味着每个顶点都将单独作为一个批次进行处理。
            // Complete() ：此方法用于等待任务完成。它会阻塞当前线程，直到 `bakeNormalJob` 中的所有任务都完成。在完成任务后，
            // 可以安全地访问和使用 Job 生成的结果数据
            bakeNormalJob.Schedule(vertexCount, 100).Complete();

            var bakedSmoothedNormals = new Vector2[vertexCount];
            bakedNormals.CopyTo(bakedSmoothedNormals);

            // Allocator.Persistent需要手动释放内存
            smoothedNormalsMap.Dispose();
            normalsNative.Dispose();
            vertrxNative.Dispose();
            tangents.Dispose();
            bakedNormals.Dispose();
            return bakedSmoothedNormals;
        }

        public static Vector3[] ComputeSmoothedNormalsV3(Mesh mesh)
        {
            Vector3[] verts = mesh.vertices;
            Vector3[] nors = mesh.normals;
            Vector4[] tans = mesh.tangents;

            int vertexCount = verts.Length;

            if (tans.Length == 0)
            {
                Debug.LogError($"{mesh.name} don't have tangents.");
                return null;
            }

            //CalcSmoothedNormals
            UnsafeParallelHashMap<Vector3, Vector3> smoothedNormalsMap = new UnsafeParallelHashMap<Vector3, Vector3>(vertexCount, Allocator.Persistent);
            for (int i = 0; i < vertexCount; i++)
            {
                if (smoothedNormalsMap.ContainsKey(verts[i]))
                {
                    smoothedNormalsMap[verts[i]] = smoothedNormalsMap[verts[i]] +
                    nors[i];
                }
                else
                {
                    smoothedNormalsMap.Add(verts[i],
                     nors[i]);
                }
            }

            //BakeSmoothedNormals
            NativeArray<Vector3> normalsNative = new NativeArray<Vector3>(nors, Allocator.Persistent);
            NativeArray<Vector3> vertrxNative = new NativeArray<Vector3>(verts, Allocator.Persistent);
            NativeArray<Vector4> tangents = new NativeArray<Vector4>(tans, Allocator.Persistent);
            NativeArray<Vector3> bakedNormals = new NativeArray<Vector3>(vertexCount, Allocator.Persistent);

            BakeNormalJobV3 bakeNormalJob = new BakeNormalJobV3(vertrxNative, normalsNative, tangents, smoothedNormalsMap, bakedNormals);
            bakeNormalJob.Schedule(vertexCount, 100).Complete();

            var bakedSmoothedNormals = new Vector3[vertexCount];
            bakedNormals.CopyTo(bakedSmoothedNormals);

            smoothedNormalsMap.Dispose();
            normalsNative.Dispose();
            vertrxNative.Dispose();
            tangents.Dispose();
            bakedNormals.Dispose();
            return bakedSmoothedNormals;
        }

        struct BakeNormalJob : IJobParallelFor
        {
            // [ReadOnly]  [WriteOnly] [NativeDisableContainerSafetyRestriction] 标记为了性能，如存在读写那么会对性能有一定影响
            // [NativeDisableContainerSafetyRestriction] 此属性用于在 Job 系统中禁用容器
            //（例如 NativeArray、UnsafeHashMap 等）的安全检查
            [ReadOnly] public NativeArray<Vector3> vertrx, normals;
            [ReadOnly] public NativeArray<Vector4> tangants;
            [NativeDisableContainerSafetyRestriction]
            [ReadOnly] public UnsafeParallelHashMap<Vector3, Vector3> smoothedNormals;
            [WriteOnly] public NativeArray<Vector2> bakedNormals;

            public BakeNormalJob(NativeArray<Vector3> vertrx,
                NativeArray<Vector3> normals,
                NativeArray<Vector4> tangents,
                UnsafeParallelHashMap<Vector3, Vector3> smoothedNormals,
                NativeArray<Vector2> bakedNormals)
            {
                this.vertrx = vertrx;
                this.normals = normals;
                this.tangants = tangents;
                this.smoothedNormals = smoothedNormals;
                this.bakedNormals = bakedNormals;
            }

            void IJobParallelFor.Execute(int index)
            {
                Vector3 smoothedNormal = smoothedNormals[vertrx[index]];

                var normalOS = normals[index].normalized;
                Vector3 tangantOS = tangants[index];
                tangantOS = tangantOS.normalized;

                // 构建附切线
                // - 如果 `tangants[index].w` 等于 1，则表示使用右手坐标系（逆时针），此时不会改变双切线的方向。
                // -如果 `tangants[index].w` 等于 - 1，则表示使用左手坐标系（顺时针），此时将翻转双切线的方向。
                // 通过乘以 `tangants[index].w`，可以确保在计算双切线时正确地处理左右手坐标系的差异
                var bitangentOS = (Vector3.Cross(normalOS, tangantOS) * tangants[index].w).normalized;

                // 构建TBN矩阵
                var tbn = new Matrix4x4(tangantOS, bitangentOS, normalOS, Vector3.zero);

                // 获取转置矩阵
                tbn = tbn.transpose;

                // 通过将平滑的法线乘以转置矩阵来获得切线空间中的法线
                var bakedNormal = OctahedronNormal(tbn.MultiplyVector(smoothedNormal).normalized);

                bakedNormals[index] = bakedNormal;
            }

            // 它将给定的法线向量（`ResultNormal`）从三维空间（`Vector3`）映射到二维空间（`Vector2`），
            // 使用了一种称为八面体（Octahedron）映射的技术。
            Vector2 OctahedronNormal(Vector3 ResultNormal)
            {
                Vector3 absVec = new Vector3(Mathf.Abs(ResultNormal.x), Mathf.Abs(ResultNormal.y), Mathf.Abs(ResultNormal.z));
                Vector2 OctNormal = (Vector2)ResultNormal / Vector3.Dot(Vector3.one, absVec);
                if (ResultNormal.z <= 0)
                {
                    float absY = Mathf.Abs(OctNormal.y);
                    float value = (1 - absY) * (OctNormal.y >= 0 ? 1 : -1);
                    OctNormal = new Vector2(value, value);
                }
                return OctNormal;
            }
           
        }

        struct BakeNormalJobV3 : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Vector3> vertrx, normals;
            [ReadOnly] public NativeArray<Vector4> tangants;
            [NativeDisableContainerSafetyRestriction]
            [ReadOnly] public UnsafeParallelHashMap<Vector3, Vector3> smoothedNormals;
            [WriteOnly] public NativeArray<Vector3> bakedNormals;

            public BakeNormalJobV3(NativeArray<Vector3> vertrx,
                NativeArray<Vector3> normals,
                NativeArray<Vector4> tangents,
                UnsafeParallelHashMap<Vector3, Vector3> smoothedNormals,
                NativeArray<Vector3> bakedNormals)
            {
                this.vertrx = vertrx;
                this.normals = normals;
                this.tangants = tangents;
                this.smoothedNormals = smoothedNormals;
                this.bakedNormals = bakedNormals;
            }

            void IJobParallelFor.Execute(int index)
            {
                Vector3 smoothedNormal = smoothedNormals[vertrx[index]];

                var normalOS = normals[index].normalized;
                Vector3 tangantOS = tangants[index];
                tangantOS = tangantOS.normalized;
                var bitangentOS = (Vector3.Cross(normalOS, tangantOS) * tangants[index].w).normalized;

                var tbn = new Matrix4x4(tangantOS, bitangentOS, normalOS, Vector3.zero);

                tbn = tbn.transpose;

                var bakedNormal = tbn.MultiplyVector(smoothedNormal).normalized;

                bakedNormals[index] = bakedNormal;
            }
        }
    }
}