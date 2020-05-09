using System;
using Unity.Mathematics;
using UnityEngine;

public class DebugRenderer : MonoBehaviour 
{
    private float width = 0.0f, height = 0.0f;
    private float invWidth = 0.0f, invHeight = 0.0f;
    private GameObject plane;

    public void Init(int width, int height, GameObject plane) 
    { 
        this.width = width; 
        this.height = height;
        this.plane = plane;
        invWidth = 1.0f / width;
        invHeight = 1.0f / height;
    }

    private GameObject[] sphere = null;
    private GameObject[] cylinder = null;

    public void DrawHand3D(Vector3[] landmarks) 
    {
        if(sphere == null) 
        { 
            float diameter = 0.25f;
            var scale = new Vector3(diameter, diameter, diameter);
            sphere = new GameObject[landmarks.Length]; 
            for(int i = 0; i < landmarks.Length; ++i)
            {
                sphere[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere[i].name = "Sphere" + i.ToString();
                sphere[i].transform.localScale = scale;
                Material mat = sphere[i].GetComponent<Renderer>().material; 
                mat.color = Color.red; 
            }
        }

        for(int i = 0; i < landmarks.Length; ++i)
        {
            var camera = Camera.main;
            var cameraPos = camera.transform.position;
            var planePos = plane.transform.position;
            var landmark = landmarks[i];
            var depth = landmark.z / 64.0f;

            float x = (width - landmark.x) * invWidth * Screen.width;
            float y = (height - landmark.y) * invHeight * Screen.height;
            float z = (cameraPos.z - planePos.z) - 0.5f +  depth;
            var pos =  camera.ScreenToWorldPoint(new Vector3(x, y, z));
            sphere[i].transform.localPosition = pos;
        }

        if(cylinder == null) 
        { 
            cylinder = new GameObject[landmarks.Length]; 
            for(int i = 0; i < landmarks.Length; ++i)
            {
                cylinder[i] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                cylinder[i].name = "Cylinder" + i.ToString();
                cylinder[i].transform.localScale = Vector3.zero;
                Material mat = cylinder[i].GetComponent<Renderer>().material;
                mat.color = Color.white;
            }
        }

        UpdateCylinder(0, 0, 1);
        UpdateCylinder(1, 1, 2);
        UpdateCylinder(2, 2, 3);
        UpdateCylinder(3, 3, 4);

        UpdateCylinder(4, 0, 5);
        UpdateCylinder(5, 5, 6);
        UpdateCylinder(6, 6, 7);
        UpdateCylinder(7, 7, 8);

        UpdateCylinder(8, 5, 9);
        UpdateCylinder(9, 9, 10);
        UpdateCylinder(10, 10, 11);
        UpdateCylinder(11, 11, 12);

        UpdateCylinder(12, 9, 13);
        UpdateCylinder(13, 13, 14);
        UpdateCylinder(14, 14, 15);
        UpdateCylinder(15, 15, 16);

        UpdateCylinder(16, 13, 17);

        UpdateCylinder(17, 0, 17);
        UpdateCylinder(18, 17, 18);
        UpdateCylinder(19, 18, 19);
        UpdateCylinder(20, 19, 20);
    }
    private void UpdateCylinder(int target, int sphere1, int sphere2) 
    {
        float diameter = 0.15f;
        var s1 = sphere[sphere1].transform.position;
        var s2 = sphere[sphere2].transform.position;
        var position = (s1 + s2) * 0.5f;
        var rotate = Quaternion.FromToRotation(Vector3.up, (s1 - s2).normalized);
        var scale = new Vector3(diameter, (s1 - s2).magnitude * 0.5f, diameter);
        cylinder[target].transform.localPosition = position;
        cylinder[target].transform.localRotation = rotate;
        cylinder[target].transform.localScale = scale;
    }

    public bool OpenHandPose()
    {
        if (sphere.Length==21)
        {
           
            if ((math.abs(Angle(sphere[1].transform.position, sphere[0].transform.position, sphere[2].transform.position)-180)<50.0f)&&
                (math.abs(Angle(sphere[2].transform.position, sphere[1].transform.position, sphere[3].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[3].transform.position, sphere[2].transform.position, sphere[4].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[6].transform.position, sphere[5].transform.position, sphere[7].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[7].transform.position, sphere[6].transform.position, sphere[8].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[10].transform.position, sphere[9].transform.position, sphere[11].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[11].transform.position, sphere[10].transform.position, sphere[12].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[14].transform.position, sphere[13].transform.position, sphere[15].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[15].transform.position, sphere[14].transform.position, sphere[16].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[18].transform.position, sphere[17].transform.position, sphere[19].transform.position) - 180) < 50.0f) &&
                (math.abs(Angle(sphere[19].transform.position, sphere[18].transform.position, sphere[20].transform.position) - 180) < 50.0f))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    }

    private double Angle(Vector3 cen, Vector3 first, Vector3 second)
    {
        double M_PI = 3.1415926535897;

        double ma_x = first.x - cen.x;
        double ma_y = first.y - cen.y;
        double ma_z = first.z - cen.z;
        double mb_x = second.x - cen.x;
        double mb_y = second.y - cen.y;
        double mb_z = second.z - cen.z;
        double v1 = (ma_x * mb_x) + (ma_y * mb_y) + (ma_z * mb_z);
        double ma_val = Math.Sqrt(ma_x * ma_x + ma_y * ma_y + ma_z * ma_z);
        double mb_val = Math.Sqrt(mb_x * mb_x + mb_y * mb_y + mb_z * mb_z);
        double cosM = v1 / (ma_val * mb_val);
        double angleAMB = Math.Acos(cosM) * 180 / M_PI;

        return angleAMB;
    }

}
