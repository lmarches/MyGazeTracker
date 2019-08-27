package com.example.mygazetracker;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;

import android.Manifest;
import android.os.Bundle;
import org.opencv.core.*;

import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.*;
import org.opencv.imgproc.Imgproc;

import android.content.pm.PackageManager;
import java.util.Random;

import org.opencv.objdetect.CascadeClassifier;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.*;



public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    private static final int MY_CAMERA_REQUEST_CODE = 100;
    private static final String TAG = "MyGazeTracker";

    int counter=0;
    private CascadeClassifier cascadeClassifierFace;
    private CascadeClassifier cascadeClassifierEye;
    private File mCascadeFileFace;
    private File mCascadeFileEye;
//    private TextView infoFaces;

    class IntPair{
        public IntPair(int i_, int j_){
            i=i_;
            j=j_;

        }
        int i;
        int j;
    };


    ArrayList<IntPair> point_trajectory = new ArrayList<IntPair>();



    private void loadHaarCascadeFile(String filenameFilterFace,String filenameFilterEye) {
        try {
//            File cascadeDir = getDir("haarcascade_frontalface_alt", Context.MODE_PRIVATE);
            mCascadeFileFace = new File(getDir("haarcascade_frontalface_alt", Context.MODE_PRIVATE), filenameFilterFace);
            mCascadeFileEye = new File(getDir("haarcascade_eye", Context.MODE_PRIVATE), filenameFilterEye);

            if (!mCascadeFileFace.exists()) {
                FileOutputStream os = new FileOutputStream(mCascadeFileFace);
                InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }

            if (!mCascadeFileEye.exists()) {
                FileOutputStream os = new FileOutputStream(mCascadeFileEye);
                InputStream is = getResources().openRawResource(R.raw.haarcascade_eye);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }

        } catch (Throwable throwable) {
            throw new RuntimeException("Failed to load Haar Cascade file");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView); //setup the camera bridge view
        cameraBridgeViewBase.setCameraIndex(1);

        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        point_trajectory = new ArrayList<IntPair>();


        point_trajectory.add(new IntPair(10,30));
        point_trajectory.add(new IntPair(100,30));
        point_trajectory.add(new IntPair(200,30));
        point_trajectory.add(new IntPair(300,30));
        point_trajectory.add(new IntPair(400,30));
        point_trajectory.add(new IntPair(500,30));
        point_trajectory.add(new IntPair(600,30));
        point_trajectory.add(new IntPair(700,30));
        point_trajectory.add(new IntPair(700,100));
        point_trajectory.add(new IntPair(10,200));
        point_trajectory.add(new IntPair(10,300));
        point_trajectory.add(new IntPair(10,400));
        point_trajectory.add(new IntPair(10,500));
        point_trajectory.add(new IntPair(10,600));
        point_trajectory.add(new IntPair(10,700));


        if (checkSelfPermission(Manifest.permission.CAMERA)

                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA},
                    MY_CAMERA_REQUEST_CODE);
        }


        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch (status) {

                    case BaseLoaderCallback.SUCCESS:


                        loadHaarCascadeFile("haarcascade_frontalface_alt.xml","haarcascade_eye.xml");


                        cascadeClassifierFace = new CascadeClassifier(mCascadeFileFace.getAbsolutePath());
                        cascadeClassifierFace.load(mCascadeFileFace.getAbsolutePath());

                        cascadeClassifierEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                        cascadeClassifierEye.load(mCascadeFileEye.getAbsolutePath());

                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }
        };


    }



    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();

        Core.rotate(frame,frame, Core.ROTATE_90_COUNTERCLOCKWISE);

//        if (counter % 10 == 0){
//            Core.flip(frame, frame, 1);
//            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2GRAY);
//        }

        int index = counter % 15;
        IntPair point = point_trajectory.get(index);

            //Drawing a Circle
         Imgproc.circle(
                    frame,
                    new Point(point.i , point.j),
                    30,
                    new Scalar(0, 0, 255),
                    30
            );

        if (frame != null) {
            MatOfRect matOfRectFaces = new MatOfRect();
            MatOfRect matOfRectEyes = new MatOfRect();
            cascadeClassifierFace.detectMultiScale(frame, matOfRectFaces);
            cascadeClassifierEye.detectMultiScale(frame, matOfRectEyes);

            int newQtdFaces = matOfRectFaces.toList().size();

            for (Rect rect : matOfRectFaces.toArray()) {
//                String log = String.format("[LUCA]:  x=%d  - y=%d - width=%d - height=%d", rect.x, rect.y,rect.width,rect.height);
//                Log.d(TAG, log);
                Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0),5);
            }

            for (Rect rect : matOfRectEyes.toArray()) {
//                String log = String.format("[LUCA]:  x=%d      - y=%d - width=%d - height=%d", rect.x, rect.y,rect.width,rect.height);
//                Log.d(TAG, log);
                Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(255, 0, 0),5);
            }

        }

        counter = counter + 1;

        return frame;
    }



    @Override
    public void onCameraViewStarted(int width, int height) {

    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}
