package com.example.mygazetracker;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;

import android.Manifest;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;

import org.opencv.android.Utils;
import org.opencv.core.*;

import android.provider.MediaStore;
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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import org.opencv.objdetect.CascadeClassifier;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.*;

import org.tensorflow.lite.Interpreter;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    private static final int MY_CAMERA_REQUEST_CODE = 100;
    private static final int MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1;

    private static final String TAG = "MyGazeTracker";

    int counter=0;
    private CascadeClassifier cascadeClassifierFace;
    private CascadeClassifier cascadeClassifierEye;
    private File mCascadeFileFace;
    private File mCascadeFileEye;
//    private TextView infoFaces;



    /////////////////////////////////////////////////////////// <---- TF  variables


    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;

    // holds all the possible labels for model
    private List<String> labelList;

    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;


    // selected classifier information received from extras
    private String chosen;
    private boolean quant=false;

    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 299;
    private int DIM_IMG_SIZE_Y = 299;
    private int DIM_PIXEL_SIZE = 3;

    // int array to hold image data
    private int[] intValues;
    /////////////////////////////////////////////////////////// <---- TF  variables

    /////////////////////////////////////////////////////////// <---- My TF  variables
    // Input placeholders
    private ByteBuffer eyeLeftData = null;
    private ByteBuffer eyeRightData = null;
    private ByteBuffer faceCropData = null;
    private ByteBuffer faceMaskData = null;
    private float[][] add_5_Data = null;

    // Output placeholders
    private static final int DIM_EYE_DATA = 64;
    private static final int DIM_FACE_CROP_DATA = 64;
    private static final int DIM_FACE_MASK_DATA = 25;
    private static final int DIM_OUT_PREDICTION = 2;

    private static final int DIM_PIXEL_SIZE_GAZE = 3;
    private static final int BYTES =4;

    // https://github.com/Rohithkvsp/MNIST-on-Android-with-TensorflowLite-and-OpenCV/blob/master/app/src/main/java/com/example/rohithkvsp/handwittendigit/Classifier.java
    private  static final float NORMALIZATION_FACTOR = 255.f;

    /////////////////////////////////////////////////////////// <---- My TF  variables



    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });


    // loads tflite grapg from file
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(chosen);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    //convert opencv mat to tensorflowlite input
    // normalize between 0 and 1
    private void convertMattoTfLiteInput(Mat mat, ByteBuffer imgTmp)
    {
        imgTmp.rewind();
        int pixel = 0;
        for (int i = 0; i < mat.height(); ++i) {
            for (int j = 0; j < mat.width(); ++j) {
                imgTmp.putFloat((float)mat.get(i,j)[0]/ NORMALIZATION_FACTOR);
            }
        }
    }




//    Bitmap bitmap_debug_1 = Bitmap.createBitmap(faceCropMat.cols(), faceCropMat.rows(), Bitmap.Config.ARGB_8888);
//    Utils.matToBitmap(faceCropMat, bitmap_debug_1);
//


    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                if(quant){
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat(((val >> 16) & 0xFF)/(float)255.0);
                    imgData.putFloat(((val >> 8) & 0xFF)/(float)255.0);
                    imgData.putFloat(((val) & 0xFF)/(float)255.0);

//                    imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                    imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                    imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }
    }


    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    class IntPair{
        public IntPair(int i_, int j_){
            i=i_;
            j=j_;

        }
        int i;
        int j;
    };

    // print the top labels and respective confidences
    private void printTopKLabels() {
        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            if(quant){
                sortedLabels.add(
                        new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArrayB[0][i] & 0xff) / 255.0f));
            } else {
                sortedLabels.add(
                        new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            }
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        // get top results from priority queue
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
        }

        // set the corresponding textviews with the results

        Log.d(TAG, "[TF-class]: 1. "+topLables[2]);
        Log.d(TAG, "[TF-class]: 2. "+topLables[1]);
        Log.d(TAG, "[TF-class]: 3. "+topLables[0]);
        Log.d(TAG, topConfidence[2]);
        Log.d(TAG, topConfidence[1]);
        Log.d(TAG, topConfidence[0]);
    }


    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }


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
        point_trajectory.add(new IntPair(700,200));
        point_trajectory.add(new IntPair(700,300));
        point_trajectory.add(new IntPair(700,400));
        point_trajectory.add(new IntPair(700,500));
        point_trajectory.add(new IntPair(700,600));
        point_trajectory.add(new IntPair(600,600));
        point_trajectory.add(new IntPair(500,500));
        point_trajectory.add(new IntPair(400,400));
        point_trajectory.add(new IntPair(300,300));
        point_trajectory.add(new IntPair(200,200));
        point_trajectory.add(new IntPair(100,100));
        point_trajectory.add(new IntPair(50,50));
        point_trajectory.add(new IntPair(0,0));
        // Get permission to use camera
        if (checkSelfPermission(Manifest.permission.CAMERA)

                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA},
                    MY_CAMERA_REQUEST_CODE);
        }

        // Get permission to write images in the gallery
        if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
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

//                        chosen = "inception_float.tflite";
//                        chosen = "inception_quant.tflite";
                        chosen = "model-23.tflite";

                        quant = true ;

                        // initialize array that holds image data
                        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

                        // initilize graph and labels
                        try{
                            tflite = new Interpreter(loadModelFile(), tfliteOptions);
//                            labelList = loadLabelList();  // TF - update:  remove the list of labels
                        } catch (Exception ex){
                            ex.printStackTrace();
                        }

                        // Initialize byte arrays for tensorflow data
                        eyeLeftData = ByteBuffer.allocateDirect(DIM_EYE_DATA*
                                                                DIM_EYE_DATA*
                                                                DIM_PIXEL_SIZE_GAZE*
                                                                BYTES);
                        eyeLeftData.order(ByteOrder.nativeOrder());

                        eyeRightData = ByteBuffer.allocateDirect(DIM_EYE_DATA*
                                                                 DIM_EYE_DATA*
                                                                 DIM_PIXEL_SIZE_GAZE*
                                                                 BYTES);
                        eyeRightData.order(ByteOrder.nativeOrder());

                        faceCropData = ByteBuffer.allocateDirect(DIM_FACE_CROP_DATA*
                                                                 DIM_FACE_CROP_DATA*
                                                                 DIM_PIXEL_SIZE_GAZE*
                                                                 BYTES);
                        faceCropData.order(ByteOrder.nativeOrder());

                        //only one channel for the mask
                        faceMaskData = ByteBuffer.allocateDirect(DIM_FACE_MASK_DATA*
                                                                 DIM_FACE_MASK_DATA*
                                                                 BYTES);
                        faceMaskData.order(ByteOrder.nativeOrder());

                        add_5_Data = new float[1][DIM_OUT_PREDICTION];


//                        // initialize byte array. The size depends if the input data needs to be quantized or not
//                        if(quant){
//                            imgData =
//                                    ByteBuffer.allocateDirect(
//                                            DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
//                        } else {
//                            imgData =
//                                    ByteBuffer.allocateDirect(
//                                            4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
//                        }
//                        imgData.order(ByteOrder.nativeOrder());
//
//                        // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
//                        if(quant){
//                            labelProbArrayB= new byte[1][labelList.size()];
//                        } else {
//                            labelProbArray = new float[1][labelList.size()];
//                        }
//
//                        // initialize array to hold top labels
//                        topLables = new String[RESULTS_TO_SHOW];
//                        // initialize array to hold top probabilities
//                        topConfidence = new String[RESULTS_TO_SHOW];

                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }
        };


    }



    public String saveToGallery(Mat imageMat,String filename){
        Bitmap tmpBitmap = Bitmap.createBitmap(imageMat.cols(),
                imageMat.rows(),
                Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(imageMat, tmpBitmap);
        // Save to gallery
        return  MediaStore.Images.Media.insertImage(getContentResolver(),
                        tmpBitmap,
                        filename,
                "inputplaceholder");
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        boolean resizeFrame = false;
//
//        resizeFrame = false;
//
        Mat frame = new Mat();
//
//        if (resizeFrame){
//            Imgproc.resize(inputFrame.rgba(),frame,new Size(360,360));
//
//        }else{
//            frame = inputFrame.rgba();
//        }
//
        frame = inputFrame.rgba();
//        Mat eyeLeftCropMatResized = new Mat();
//        Imgproc.resize(eyeLeftCropMat,eyeLeftCropMatResized,sz_eyes);

        Core.rotate(frame,frame, Core.ROTATE_90_COUNTERCLOCKWISE);

//        // Test the camera by flipping image and converting to black and white
//        if (counter % 10 == 0){
//            Core.flip(frame, frame, 1);
//            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2GRAY);
//        }

        // specifies the number of frames skipped in the processing pipeline
        int index = counter % 1;
        int index_target = counter %20;

        if (frame != null & index == 0 ) {

//            //converts RGB frame into Bitmap
//            Bitmap bitmap_orig = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
//            Utils.matToBitmap(frame, bitmap_orig);
//
//            // get current bitmap from imageView
////            Bitmap bitmap_orig = ((BitmapDrawable)selected_image.getDrawable()).getBitmap();
//            // resize the bitmap to the required input size to the CNN
//            Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
//            // convert bitmap to byte array
//            convertBitmapToByteBuffer(bitmap);

//            tflite.runForMultipleInputsOutputs();
//
            // TF - update
//            // pass byte data to the graph
//            if(quant){
//                tflite.run(imgData, labelProbArrayB);
//            } else {
//                tflite.run(imgData, labelProbArray);
//            }
//
//            printTopKLabels();
            // TF - update


            int xPix = 0;
            int yPix = 0;


            // detect faces
            MatOfRect matOfRectFaces = new MatOfRect();
            cascadeClassifierFace.detectMultiScale(frame, matOfRectFaces);
            int newQtdFaces = matOfRectFaces.toList().size();

            // Loop through the faces detected in image
            for (Rect rectFace : matOfRectFaces.toArray()) {
                String log = String.format("[LUCA]:  x=%d  - y=%d - width=%d - height=%d", rectFace.x, rectFace.y,rectFace.width,rectFace.height);
                Log.d(TAG, log);

                // crop the face sub image image by first specifying the rectangle to carve out
                Rect faceCrop = new Rect(rectFace.x, rectFace.y, rectFace.width, rectFace.height);
                // Get the cropped face into a new image
                Mat faceCropMat = frame.submat(faceCrop);

                //converts RGB frame into Bitmap

//                Bitmap bitmap_debug_1 = Bitmap.createBitmap(faceCropMat.cols(), faceCropMat.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(faceCropMat, bitmap_debug_1);

                    //Initialise rectangles for the eyes
                MatOfRect matOfRectEyes = new MatOfRect();

                // detect eyes
                cascadeClassifierEye.detectMultiScale(faceCropMat, matOfRectEyes);


                Rect[] eyesArray = matOfRectEyes.toArray();
                boolean checkPassed=false;

                // check if we have two eyes
                if (eyesArray.length >1){
                    // check if size of eyes is within norm
//                    for (Rect rectEyes : matOfRectEyes.toArray()) {
//
//
//                    }
                    checkPassed=true;

                }

                // you have one face, at least two eyes, try to predict gaze
                if (checkPassed) {

                    // Step 1: gather all the input images
                    // get the crop for the first eye ( eyesArray[0] )
                    Rect eyeLeftCropRect = new Rect(eyesArray[0].x, eyesArray[0].y,
                                                eyesArray[0].width, eyesArray[0].height);
                    Mat eyeLeftCropMat = faceCropMat.submat(eyeLeftCropRect);

                    // get the crop for the second eye
                    Rect eyeRightCropRect = new Rect(eyesArray[1].x,eyesArray[1].y,
                                                    eyesArray[1].width, eyesArray[1].height);
                    Mat eyeRightCropMat = faceCropMat.submat(eyeRightCropRect);

                    // create a binary mask containing the face bounding box in white
                    Mat faceMaskMat = Mat.zeros(frame.width(),frame.height(),CvType.CV_8U);
                    Imgproc.rectangle(faceMaskMat,
                                        new Point(rectFace.x,rectFace.y),
                                        new Point(rectFace.x +rectFace.width, rectFace.y + rectFace.height),
                                        new Scalar(255),
                                        -1,
                                        1,
                                        0);



                    // Step 2: resize everything according to specs
                    //
                    // eye_left 64x64
                    // eye_right 64x64
                    // face 64x64
                    // face_mask 625
                    // Add_5 shape=(?, 2) dtype=float32>

                    Size sz_eyes = new Size(64,64);
                    Size sz_face = new Size(64,64);
                    Size sz_faceMask = new Size(25,25);

                    Mat eyeLeftCropMatResized = new Mat();
                    Imgproc.resize(eyeLeftCropMat,eyeLeftCropMatResized,sz_eyes);

                    Mat eyeRightCropMatResized = new Mat();
                    Imgproc.resize(eyeRightCropMat, eyeRightCropMatResized, sz_eyes);

                    Mat faceCropMatResized = new Mat();
                    Imgproc.resize(faceCropMat,faceCropMatResized,sz_face);

                    Mat faceMaskMatResized = new Mat();
                    Imgproc.resize(faceMaskMat,faceMaskMatResized,sz_faceMask);


                    saveToGallery(eyeLeftCropMat,Integer.toString(index)+"_eyeLeft.png");
                    saveToGallery(eyeRightCropMat,Integer.toString(index)+"_eyeRight.png");
                    saveToGallery(faceCropMatResized,Integer.toString(index)+"_face.png");
                    saveToGallery(faceMaskMatResized,Integer.toString(index)+"_faceMask.png");


                    // Step 3. Convert to ByteBuffer


                    faceMaskMatResized.reshape(1,faceMaskMatResized.width()*faceMaskMatResized.height());

                    convertMattoTfLiteInput(eyeLeftCropMatResized, eyeLeftData);
                    convertMattoTfLiteInput(eyeRightCropMatResized, eyeRightData);
                    convertMattoTfLiteInput(faceCropMatResized, faceCropData);
                    convertMattoTfLiteInput(faceMaskMatResized, faceMaskData);


                    // Step 4. Concatenate I/O placeholders

                    // From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java
                    // Object[] inputs = {input0, input1, ...};
                    Object[] tf_inputs = {eyeLeftData,eyeRightData,faceCropData,faceMaskData};

                    // example in https://stackoverflow.com/questions/54688482/how-to-allocate-multi-dimensional-outputbuffer-to-feed-android-tflites-interpre/56643983
                    Map<Integer, Object> tf_outputs  = new HashMap<>();
                    tf_outputs.put(0,add_5_Data);


                    // Step 5. Run TF session
                    tflite.runForMultipleInputsOutputs(tf_inputs,tf_outputs);

                    float[][] predictions = (float[][]) tf_outputs.get(0);

                    xPix = (int)(720*((predictions[0][0]-0.45)/(1.1-0.45)));
                    yPix = (int)(((360.0/.4)*predictions[0][1])+360);

                    log = String.format("[OUTPUT]: %f ,%f,%d ,%d",predictions[0][0], predictions[0][1],xPix, yPix);
                    Log.d(TAG, log);

                    // TF - update
//            // pass byte data to the graph
//            if(quant){
//                tflite.run(imgData, labelProbArrayB);.0
//            } else {
//                tflite.run(imgData, labelProbArray);
//            }
//
//            printTopKLabels();
                    // TF - update
                    // draw bounding boxes
                    for (Rect rectEyes : matOfRectEyes.toArray()) {

                        log = String.format("[LUCA]: Face width=%d - height=%d",faceCropMat.width(), faceCropMat.height());
                        Log.d(TAG, log);


                        log = String.format("[LUCA]: Eyes coord x=%d - y=%d - width=%d - height=%d", rectEyes.x, rectEyes.y,rectEyes.width,rectEyes.height);
                        Log.d(TAG, log);

                        // draw eyes bounding boxes into frame
                        Imgproc.rectangle(frame, new Point(rectFace.x+rectEyes.x, rectFace.y+rectEyes.y),
                                new Point(rectFace.x+rectEyes.x +rectEyes.width, rectFace.y+rectEyes.y + rectEyes.height),
                                new Scalar(255, 0, 0),5);

                    }
                }

                // draw faces bounding box
                Imgproc.rectangle(frame, new Point(rectFace.x, rectFace.y),
                        new Point(rectFace.x + rectFace.width, rectFace.y + rectFace.height),
                        new Scalar(0, 255, 0),5);
            }


            IntPair point = point_trajectory.get(index_target);


            //Drawing a Circle
            Imgproc.circle(
                    frame,
                    new Point(xPix , yPix),
                    30,
                    new Scalar(0, 0, 255),
                    30
            );

            Imgproc.circle(
                    frame,
                    new Point(point.i,point.j),
                    30,
                    new Scalar(255,0,0),
                    30
            );
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
