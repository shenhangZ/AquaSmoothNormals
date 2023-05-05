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
            // ����һ�������ܡ��̰߳�ȫ�� hashmap ���ݽṹ���ر������ڶ��̲߳��м��㳡����
            UnsafeParallelHashMap<Vector3, Vector3> smoothedNormalsMap = new UnsafeParallelHashMap<Vector3, Vector3>(vertexCount, Allocator.Persistent);
            for (int i = 0; i < vertexCount; i++)
            {
                // �ظ�������������ӷ���ֵ
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
            //`Allocator.Persistent` ������������������ִ���ڼ�־÷����ڴ档���仰˵����ʹ�� `Allocator.Persistent` �����ڴ�ʱ
            //������ڴ潫����������������������Զ��ͷš����ֲ���ͨ�����ڷ�����Ҫ����������ִ���ڼ�һֱ���ڵ���Դ��
            NativeArray<Vector3> normalsNative = new NativeArray<Vector3>(normals, Allocator.Persistent);
            NativeArray<Vector3> vertrxNative = new NativeArray<Vector3>(vertices, Allocator.Persistent);
            NativeArray<Vector4> tangents = new NativeArray<Vector4>(tans, Allocator.Persistent);
            NativeArray<Vector2> bakedNormals = new NativeArray<Vector2>(vertexCount, Allocator.Persistent);

            BakeNormalJob bakeNormalJob = new BakeNormalJob(vertrxNative, normalsNative, tangents, smoothedNormalsMap, bakedNormals);
            // Schedule() �������ڽ�Ҫִ�е�������ӵ� Job ϵͳ
            // vertexCount ����ʾ������ֳɶ��ٸ����Σ�����ÿ�����ΰ���һ�����㡣����ζ��ÿ�����㶼��������Ϊһ�����ν��д���
            // Complete() ���˷������ڵȴ�������ɡ�����������ǰ�̣߳�ֱ�� `bakeNormalJob` �е�����������ɡ�����������
            // ���԰�ȫ�ط��ʺ�ʹ�� Job ���ɵĽ������
            bakeNormalJob.Schedule(vertexCount, 100).Complete();

            var bakedSmoothedNormals = new Vector2[vertexCount];
            bakedNormals.CopyTo(bakedSmoothedNormals);

            // Allocator.Persistent��Ҫ�ֶ��ͷ��ڴ�
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
            // [ReadOnly]  [WriteOnly] [NativeDisableContainerSafetyRestriction] ���Ϊ�����ܣ�����ڶ�д��ô���������һ��Ӱ��
            // [NativeDisableContainerSafetyRestriction] ������������ Job ϵͳ�н�������
            //������ NativeArray��UnsafeHashMap �ȣ��İ�ȫ���
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

                // ����������
                // - ��� `tangants[index].w` ���� 1�����ʾʹ����������ϵ����ʱ�룩����ʱ����ı�˫���ߵķ���
                // -��� `tangants[index].w` ���� - 1�����ʾʹ����������ϵ��˳ʱ�룩����ʱ����ת˫���ߵķ���
                // ͨ������ `tangants[index].w`������ȷ���ڼ���˫����ʱ��ȷ�ش�������������ϵ�Ĳ���
                var bitangentOS = (Vector3.Cross(normalOS, tangantOS) * tangants[index].w).normalized;

                // ����TBN����
                var tbn = new Matrix4x4(tangantOS, bitangentOS, normalOS, Vector3.zero);

                // ��ȡת�þ���
                tbn = tbn.transpose;

                // ͨ����ƽ���ķ��߳���ת�þ�����������߿ռ��еķ���
                var bakedNormal = OctahedronNormal(tbn.MultiplyVector(smoothedNormal).normalized);

                bakedNormals[index] = bakedNormal;
            }

            // ���������ķ���������`ResultNormal`������ά�ռ䣨`Vector3`��ӳ�䵽��ά�ռ䣨`Vector2`����
            // ʹ����һ�ֳ�Ϊ�����壨Octahedron��ӳ��ļ�����
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