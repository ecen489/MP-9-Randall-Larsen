// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.firebase.codelab.mlkit;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import com.google.android.gms.tasks.Continuation;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.codelab.mlkit.GraphicOverlay.Graphic;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
//import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.model.FirebaseCloudModelSource;
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource;
import com.google.firebase.ml.custom.model.FirebaseModelDownloadConditions;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText;
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentTextRecognizer;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import android.os.SystemClock;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.Interpreter;
import android.content.res.AssetFileDescriptor;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;


public  class MainActivity extends AppCompatActivity implements MyRecyclerViewAdapter.ItemClickListener {
    private static final String TAG = "MainActivity";
    private ImageView mImageView;
    private Button mTextButton;
    private Button mFaceButton;
    private Button mCloudButton;
    private Button mRunCustomModelButton;
    private Bitmap mSelectedImage;
    static Bitmap disp;
    private GraphicOverlay mGraphicOverlay;
    // Max width (portrait mode)
    private Integer mImageMaxWidth;
    // Max height (portrait mode)

    private Integer mImageMaxHeight;
    /**
     * Name of the model file hosted with Firebase.
     */
    private static final String HOSTED_MODEL_NAME = "cloud_model_1";
    private static final String LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite";
    /**
     * Name of the label file stored in Assets.
     */
    private static final String LABEL_PATH = "labels.txt";
    /**
     * Number of results to show in the UI.
     */
    private static final int RESULTS_TO_SHOW = 3;
    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int DIM_IMG_SIZE_X = 224;
    private static final int DIM_IMG_SIZE_Y = 224;
    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> mLabelList;

    private final PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float>
                                o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });
    /* Preallocated buffers for storing image data. */
    private final int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    /**
     * An instance of the driver class to run model inference with Firebase.
     */
    //private FirebaseModelInterpreter mInterpreter;
    /**
     * Data configuration of input & output data of model.
     */
    private FirebaseModelInputOutputOptions mDataOptions;


    private static final int CAMERA_REQUEST = 1888;
    ArrayList<String> animalNames = new ArrayList<>();
    private Interpreter tflite;
    private List<String> labelList;
    private float[][] labelProbArray = null;
    private ByteBuffer imgData = null;
    private static final String MODEL_PATH = "graph.lite";

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private float[][] filterLabelProbArray = null;
    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR = 0.4f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            tflite = new Interpreter(loadModelFile(getParent()));
           labelList = loadLabelList(getParent());
            imgData =
                    ByteBuffer.allocateDirect(
                            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            imgData.order(ByteOrder.nativeOrder());
            labelProbArray = new float[1][labelList.size()];
            filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
        } catch (IOException e) {
            e.printStackTrace();
        }

        mImageView = findViewById(R.id.image_view);

        mTextButton = findViewById(R.id.button_text);

        mTextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runTextRecognition();
            }
        });



        String[] items = new String[]{"Test Image 1 (Text)", "Test Image 2 (Text)", "Test Image 3" +
                " (Face)", "Test Image 4 (Object)", "Test Image 5 (Object)"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout
                .simple_spinner_dropdown_item, items);

       // initCustomModel();

        Button Button = (Button) this.findViewById(R.id.cam_but);

        Button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
        });
    }

    public byte[] getBytesFromBitmap(Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        return stream.toByteArray();
    }


    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CAMERA_REQUEST) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");

            Matrix matrix = new Matrix();
            // matrix.preScale(-1.0f, 1.0f);
            matrix.preRotate(90);
            Bitmap bOutput = Bitmap.createBitmap(photo, 0, 0, photo.getWidth(), photo.getHeight(), matrix, true);
            disp   = Bitmap.createScaledBitmap(bOutput,224,224,false);;
            mImageView.setImageBitmap(bOutput);
            // classifyFrame(bOutput);
            animalNames.add(""+classifyFrame(disp));

            RecyclerView recyclerView = findViewById(R.id.rvAnimals);
            recyclerView.setLayoutManager(new LinearLayoutManager(MainActivity.this));
            MyRecyclerViewAdapter adapter = new MyRecyclerViewAdapter(MainActivity.this, animalNames);
            adapter.setClickListener(MainActivity.this);
            recyclerView.setAdapter(adapter);
        }
    }



    /** Prints top-K labels, to be shown in UI as the results. */
    private String printTopKLabels() {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = String.format("\n%s: %4.2f",label.getKey(),label.getValue()) + textToShow;
        }
        return textToShow;
    }

    String classifyFrame(Bitmap bitmap) {

        convertBitmapToByteBuffer(bitmap);
        // Here's where the magic happens!!!

        tflite.run(imgData, labelProbArray);



        // smooth the results
        applyFilter();

        // print the results
        String textToShow = printTopKLabels();

        return textToShow;
    }
    void applyFilter(){
        int num_labels =  labelList.size();

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for(int j=0; j<num_labels; ++j){
            filterLabelProbArray[0][j] += FILTER_FACTOR*(labelProbArray[0][j] -
                    filterLabelProbArray[0][j]);
        }
        // Low pass filter each stage into the next.
        for (int i=1; i<FILTER_STAGES; ++i){
            for(int j=0; j<num_labels; ++j){
                filterLabelProbArray[i][j] += FILTER_FACTOR*(
                        filterLabelProbArray[i-1][j] -
                                filterLabelProbArray[i][j]);

            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for(int j=0; j<num_labels; ++j){
            labelProbArray[0][j] = filterLabelProbArray[FILTER_STAGES-1][j];
        }
    }

    private void runTextRecognition() {
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(disp);
        FirebaseVisionTextRecognizer recognizer = FirebaseVision.getInstance()
                .getOnDeviceTextRecognizer();
        mTextButton.setEnabled(false);
        recognizer.processImage(image)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseVisionText>() {
                            @Override
                            public void onSuccess(FirebaseVisionText texts) {
                                mTextButton.setEnabled(true);
                                processTextRecognitionResult(texts);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                mTextButton.setEnabled(true);
                                e.printStackTrace();
                            }
                        });
    }

    private void processTextRecognitionResult(FirebaseVisionText texts) {
        List<FirebaseVisionText.TextBlock> blocks = texts.getTextBlocks();
        if (blocks.size() == 0) {
            showToast("No text found");
            return;
        }
        animalNames.add(""+texts.getText() );
        //mGraphicOverlay.clear();
        for (int i = 0; i < blocks.size(); i++) {
            List<FirebaseVisionText.Line> lines = blocks.get(i).getLines();

            for (int j = 0; j < lines.size(); j++) {
                List<FirebaseVisionText.Element> elements = lines.get(j).getElements();
                for (int k = 0; k < elements.size(); k++) {

                }
            }

        }
        RecyclerView recyclerView = findViewById(R.id.rvAnimals);
        recyclerView.setLayoutManager(new LinearLayoutManager(MainActivity.this));
        MyRecyclerViewAdapter adapter = new MyRecyclerViewAdapter(MainActivity.this, animalNames);
        adapter.setClickListener(MainActivity.this);
        recyclerView.setAdapter(adapter);
    }

    private void runFaceContourDetection() {
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(mSelectedImage);
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                        .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                        .build();

        mFaceButton.setEnabled(false);
        FirebaseVisionFaceDetector detector = FirebaseVision.getInstance().getVisionFaceDetector(options);
        detector.detectInImage(image)
                .addOnSuccessListener(
                        new OnSuccessListener<List<FirebaseVisionFace>>() {
                            @Override
                            public void onSuccess(List<FirebaseVisionFace> faces) {
                                mFaceButton.setEnabled(true);
                                processFaceContourDetectionResult(faces);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                mFaceButton.setEnabled(true);
                                e.printStackTrace();
                            }
                        });

    }

    private void processFaceContourDetectionResult(List<FirebaseVisionFace> faces) {
        // Task completed successfully
        if (faces.size() == 0) {
            showToast("No face found");
            return;
        }
        mGraphicOverlay.clear();
        for (int i = 0; i < faces.size(); ++i) {
            FirebaseVisionFace face = faces.get(i);
            FaceContourGraphic faceGraphic = new FaceContourGraphic(mGraphicOverlay);
            mGraphicOverlay.add(faceGraphic);
            faceGraphic.updateFace(face);
        }
    }

    /*private void initCustomModel() {
        mLabelList = loadLabelList(this);

        int[] inputDims = {DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE};
        int[] outputDims = {DIM_BATCH_SIZE, mLabelList.size()};
        try {
            mDataOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                            .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                            .build();
            FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions
                    .Builder()
                    .requireWifi()
                    .build();
            FirebaseCloudModelSource cloudSource = new FirebaseCloudModelSource.Builder
                    (HOSTED_MODEL_NAME)
                    .enableModelUpdates(true)
                    .setInitialDownloadConditions(conditions)
                    .setUpdatesDownloadConditions(conditions)  // You could also specify
                    // different conditions
                    // for updates
                    .build();
            FirebaseLocalModelSource localSource =
                    new FirebaseLocalModelSource.Builder("asset")
                            .setAssetFilePath(LOCAL_MODEL_ASSET).build();
            FirebaseModelManager manager = FirebaseModelManager.getInstance();
            manager.registerCloudModelSource(cloudSource);
            manager.registerLocalModelSource(localSource);
            FirebaseModelOptions modelOptions =
                    new FirebaseModelOptions.Builder()
                            .setCloudModelName(HOSTED_MODEL_NAME)
                            .setLocalModelName("asset")
                            .build();
            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions);
        } catch (FirebaseMLException e) {
            showToast("Error while setting up the model");
            e.printStackTrace();
        }
    }*/

   /* private void runModelInference() {
        if (mInterpreter == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return;
        }
        // Create input data.
        ByteBuffer imgData = convertBitmapToByteBuffer(mSelectedImage, mSelectedImage.getWidth(),
                mSelectedImage.getHeight());

        try {
            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();
            // Here's where the magic happens!!
            mInterpreter
                    .run(inputs, mDataOptions)
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            e.printStackTrace();
                            showToast("Error running model inference");
                        }
                    })
                    .continueWith(
                            new Continuation<FirebaseModelOutputs, List<String>>() {
                                @Override
                                public List<String> then(Task<FirebaseModelOutputs> task) {
                                    byte[][] labelProbArray = task.getResult()
                                            .<byte[][]>getOutput(0);
                                    List<String> topLabels = getTopLabels(labelProbArray);
                                    mGraphicOverlay.clear();
                                    GraphicOverlay.Graphic labelGraphic = new LabelGraphic
                                            (mGraphicOverlay, topLabels);
                                    mGraphicOverlay.add(labelGraphic);
                                    return topLabels;
                                }
                            });
        } catch (FirebaseMLException e) {
            e.printStackTrace();
            showToast("Error running model inference");
        }

    }*/

    private void runCloudTextRecognition() {
        mCloudButton.setEnabled(false);
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(mSelectedImage);
        FirebaseVisionDocumentTextRecognizer recognizer = FirebaseVision.getInstance()
                .getCloudDocumentTextRecognizer();
        recognizer.processImage(image)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseVisionDocumentText>() {
                            @Override
                            public void onSuccess(FirebaseVisionDocumentText texts) {
                                mCloudButton.setEnabled(true);
                                processCloudTextRecognitionResult(texts);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                mCloudButton.setEnabled(true);
                                e.printStackTrace();
                            }
                        });
    }

    private void processCloudTextRecognitionResult(FirebaseVisionDocumentText text) {
        // Task completed successfully
        if (text == null) {
            showToast("No text found");
            return;
        }
        mGraphicOverlay.clear();
        List<FirebaseVisionDocumentText.Block> blocks = text.getBlocks();
        for (int i = 0; i < blocks.size(); i++) {
            List<FirebaseVisionDocumentText.Paragraph> paragraphs = blocks.get(i).getParagraphs();
            for (int j = 0; j < paragraphs.size(); j++) {
                List<FirebaseVisionDocumentText.Word> words = paragraphs.get(j).getWords();
                for (int l = 0; l < words.size(); l++) {
                    CloudTextGraphic cloudDocumentTextGraphic = new CloudTextGraphic(mGraphicOverlay,
                            words.get(l));
                    mGraphicOverlay.add(cloudDocumentTextGraphic);
                }
            }
        }
    }

    /**
     * Gets the top labels in the results.
     */
    private synchronized List<String> getTopLabels(byte[][] labelProbArray) {
        for (int i = 0; i < mLabelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(mLabelList.get(i), (labelProbArray[0][i] &
                            0xff) / 255.0f));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        List<String> result = new ArrayList<>();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            result.add(label.getKey() + ":" + label.getValue());
        }
        Log.d(TAG, "labels: " + result.toString());
        return result;
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(getApplication().getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        Log.e("main","ERROR file not read");
        AssetFileDescriptor fileDescriptor = getApplication().getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
   /* private synchronized ByteBuffer convertBitmapToByteBuffer(
            Bitmap bitmap) {
        ByteBuffer imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
                true);
        imgData.rewind();
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0,
                scaledBitmap.getWidth(), scaledBitmap.getHeight());
        // Convert the image to int points.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
        return imgData;
    }*/
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;

        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }

    }

    private void showToast(String message) {
        Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
    }

    // Functions for loading images from app assets.

    // Returns max image width, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxWidth() {
        if (mImageMaxWidth == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxWidth = mImageView.getWidth();
        }

        return mImageMaxWidth;
    }

    // Returns max image height, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxHeight() {
        if (mImageMaxHeight == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxHeight =
                    mImageView.getHeight();
        }

        return mImageMaxHeight;
    }

    // Gets the targeted width / height.
    private Pair<Integer, Integer> getTargetedWidthHeight() {
        int targetWidth;
        int targetHeight;
        int maxWidthForPortraitMode = getImageMaxWidth();
        int maxHeightForPortraitMode = getImageMaxHeight();
        targetWidth = maxWidthForPortraitMode;
        targetHeight = maxHeightForPortraitMode;
        return new Pair<>(targetWidth, targetHeight);
    }

    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        mGraphicOverlay.clear();
        switch (position) {
            case 0:
                mSelectedImage = getBitmapFromAsset(this, "Please_walk_on_the_grass.jpg");
                break;
            case 1:
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "nl2.jpg");
                break;
            case 2:
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "grace_hopper.jpg");
                break;
            case 3:
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "tennis.jpg");
                break;
            case 4:
                // Whatever you want to happen when the thrid item gets selected
                mSelectedImage = getBitmapFromAsset(this, "mountain.jpg");
                break;
        }
        if (mSelectedImage != null) {
            // Get the dimensions of the View
            Pair<Integer, Integer> targetedSize = getTargetedWidthHeight();

            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;

            // Determine how much to scale down the image
            float scaleFactor =
                    Math.max(
                            (float) mSelectedImage.getWidth() / (float) targetWidth,
                            (float) mSelectedImage.getHeight() / (float) maxHeight);

            Bitmap resizedBitmap =
                    Bitmap.createScaledBitmap(
                            mSelectedImage,
                            (int) (mSelectedImage.getWidth() / scaleFactor),
                            (int) (mSelectedImage.getHeight() / scaleFactor),
                            true);

            mImageView.setImageBitmap(resizedBitmap);
            mSelectedImage = resizedBitmap;
        }
    }



    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream is;
        Bitmap bitmap = null;
        try {
            is = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap;
    }

    @Override
    public void onItemClick(View view, int position) {

    }
}
