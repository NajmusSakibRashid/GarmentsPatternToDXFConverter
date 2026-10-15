package com.example.garmentpatterntodxfconverter;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.graphics.RectF;
import android.os.Bundle;
import android.text.InputType;
import android.util.Size;
import android.util.SizeF;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.ViewConfiguration;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.garmentpatterntodxfconverter.opticalflow.FlowAnalyzer;
import com.example.garmentpatterntodxfconverter.opticalflow.OpticalFlowTracker;
import com.example.garmentpatterntodxfconverter.opticalflow.Track;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    /** Weight of each new sensor sample; lower is smoother but laggier. */
    private static final float SMOOTHING = 0.4f;
    private static final float DEFAULT_VERTICAL_FOV_DEG = 65f;

    private static final int AXIS_EAST = 0;
    private static final int AXIS_NORTH = 1;
    private static final int AXIS_UP = 2;

    private static final int ANALYSIS_WIDTH = 640;
    private static final int ANALYSIS_HEIGHT = 480;

    private PreviewView previewView;
    private GroundGridView gridView;
    private FlowOverlayView flowOverlay;
    private TextView flowToggle;
    private TextView flowStatus;

    /** Created only once the native library is up; constructing it allocates OpenCV Mats. */
    private OpticalFlowTracker flowTracker;
    private FlowAnalyzer flowAnalyzer;
    private ExecutorService analysisExecutor;
    private boolean openCvReady;

    private SensorManager sensorManager;
    private Sensor rotationSensor;

    private final float[] quaternion = new float[4];
    private final float[] smoothed = new float[4];
    private boolean hasSmoothed;

    /** Which world axes are currently held still, indexed by AXIS_*. */
    private final boolean[] axisLocked = new boolean[3];

    private final float[] previousSensor = new float[4];
    private boolean hasPreviousSensor;
    /** World-frame rotation absorbed while axes were locked; persists after release. */
    private final float[] correction = {1f, 0f, 0f, 0f};
    private final float[] delta = new float[4];
    private final float[] twist = new float[4];
    private final float[] scratchA = new float[4];
    private final float[] scratchB = new float[4];
    private final float[] display = new float[4];

    private final float[] rotationMatrix = new float[9];
    private final float[] displayMatrix = new float[9];

    private int remapAxisX = SensorManager.AXIS_X;
    private int remapAxisY = SensorManager.AXIS_Y;
    private boolean needsRemap;

    private float lastTouchX;
    private float lastTouchY;
    private boolean needsTouchReseed;

    private enum Mode { NORMAL, SELECTING, PAIRING }

    private Mode mode = Mode.NORMAL;
    private final GroundAnchor groundAnchor = new GroundAnchor();
    private TextView anchorToggle;
    private TextView pairToggle;
    private TextView lockToggle;

    private final RectF selectionBox = new RectF();
    private float boxStartX;
    private float boxStartY;

    /** First anchor of a pair being measured, or -1 when nothing is half-picked. */
    private int pendingPairId = -1;

    private CameraControl cameraControl;
    /** A touch stays a tap, and so a focus request, until it travels past the slop. */
    private boolean touchIsTap;
    private float touchDownX;
    private float touchDownY;
    private int touchSlop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        gridView = findViewById(R.id.gridView);
        flowOverlay = findViewById(R.id.flowOverlay);
        flowToggle = findViewById(R.id.flowToggle);
        flowStatus = findViewById(R.id.flowStatus);
        anchorToggle = findViewById(R.id.anchorToggle);
        pairToggle = findViewById(R.id.pairToggle);
        lockToggle = findViewById(R.id.lockToggle);

        touchSlop = ViewConfiguration.get(this).getScaledTouchSlop();
        setUpOpticalFlow();
        setUpAnchoring();
        bindAxisButton(R.id.lockX, AXIS_EAST);
        bindAxisButton(R.id.lockY, AXIS_NORTH);
        bindAxisButton(R.id.lockZ, AXIS_UP);
        findViewById(R.id.reset).setOnClickListener(view -> resetGrid());
        bindGridDragging();

        gridView.setVerticalFov(readPreviewVerticalFov());

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        if (rotationSensor == null) {
            rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        }
        if (rotationSensor == null) {
            Toast.makeText(this, "No rotation sensor on this device", Toast.LENGTH_LONG).show();
        }
        resolveDisplayRemap();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    /**
     * Optical flow is gated behind the native OpenCV library loading. The listener fires on the
     * analysis thread, so the overlay is handed the data directly and the text readout, which
     * touches views, is posted to the main thread.
     */
    private void setUpOpticalFlow() {
        openCvReady = OpenCVLoader.initLocal();
        if (!openCvReady) {
            flowToggle.setEnabled(false);
            flowToggle.setAlpha(0.4f);
            flowStatus.setText("OpenCV unavailable");
            return;
        }

        flowTracker = new OpticalFlowTracker();
        analysisExecutor = Executors.newSingleThreadExecutor();
        flowAnalyzer = new FlowAnalyzer(flowTracker, (tracks, imageWidth, imageHeight) -> {
            flowOverlay.setTracks(tracks, imageWidth, imageHeight);
            int count = tracks.size();
            // The anchor update ray-casts through view state the main thread owns, so it has to
            // run there rather than on the analysis thread. The track list is immutable.
            flowOverlay.post(() -> {
                groundAnchor.update(tracks, flowOverlay, gridView);
                flowStatus.setText(describeStatus(count));
            });
        });

        flowToggle.setOnClickListener(view -> {
            boolean enable = !flowAnalyzer.isEnabled();
            flowAnalyzer.setEnabled(enable);
            flowToggle.setActivated(enable);
            if (!enable) {
                flowOverlay.clear();
                flowStatus.setText("");
            }
        });
    }

    /**
     * ANCHOR stages points, PAIR measures known distances between them, and LOCK engages
     * translation. They are separate steps because the staged points lie on a plane whose height
     * is unknown until at least one pair has been measured.
     */
    private void setUpAnchoring() {
        anchorToggle.setOnClickListener(view -> {
            if (groundAnchor.hasAnchors()) {
                groundAnchor.release();
                flowOverlay.setSelectedIds(Collections.emptySet());
                clearPairing();
            } else {
                setMode(mode == Mode.SELECTING ? Mode.NORMAL : Mode.SELECTING);
            }
            refreshModeUi();
        });

        pairToggle.setOnClickListener(view -> {
            if (!groundAnchor.hasAnchors()) {
                Toast.makeText(this, "Stage anchors first", Toast.LENGTH_SHORT).show();
                return;
            }
            setMode(mode == Mode.PAIRING ? Mode.NORMAL : Mode.PAIRING);
            pendingPairId = -1;
            flowOverlay.setPairs(groundAnchor.getPairs(), -1);
            refreshModeUi();
        });

        lockToggle.setOnClickListener(view -> {
            if (!groundAnchor.hasPairs()) {
                Toast.makeText(this, "Measure at least one pair first", Toast.LENGTH_SHORT).show();
                return;
            }
            boolean lock = !groundAnchor.isLocked();
            if (lock) {
                groundAnchor.rebaseline(flowOverlay, gridView);
            }
            groundAnchor.setLocked(lock);
            refreshModeUi();
        });

        refreshModeUi();
    }

    private void setMode(Mode next) {
        mode = next;
        if (next != Mode.SELECTING) {
            flowOverlay.setSelectionBox(null);
        }
    }

    private void clearPairing() {
        pendingPairId = -1;
        flowOverlay.setPairs(groundAnchor.getPairs(), -1);
        setMode(Mode.NORMAL);
    }

    private void refreshModeUi() {
        anchorToggle.setActivated(mode == Mode.SELECTING || groundAnchor.hasAnchors());
        pairToggle.setActivated(mode == Mode.PAIRING || groundAnchor.hasPairs());
        pairToggle.setAlpha(groundAnchor.hasAnchors() ? 1f : 0.4f);
        lockToggle.setActivated(groundAnchor.isLocked());
        lockToggle.setAlpha(groundAnchor.hasPairs() ? 1f : 0.4f);
    }

    /** Collects every tracked point whose mapped position falls inside the drawn box. */
    private void stageAnchorsInBox() {
        List<Track> tracks = flowTracker == null
                ? Collections.emptyList() : flowTracker.getTracks();
        Set<Integer> selected = new HashSet<>();
        float[] point = new float[2];
        for (Track track : tracks) {
            if (flowOverlay.imageToView(track.currentX, track.currentY, point)
                    && selectionBox.contains(point[0], point[1])) {
                selected.add(track.id);
            }
        }

        if (selected.size() < 3) {
            Toast.makeText(this, "Need at least 3 points - try a bigger box",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        groundAnchor.stage(tracks, selected, flowOverlay, gridView);
        flowOverlay.setSelectedIds(groundAnchor.getAnchorIds());
        setMode(Mode.NORMAL);
        Toast.makeText(this, groundAnchor.getAnchorCount() + " anchors staged",
                Toast.LENGTH_SHORT).show();
        refreshModeUi();
    }

    /**
     * Asks for the real separation of the two picked anchors. Each measured pair is another
     * independent estimate of the plane height, which the tracker keeps re-solving every frame.
     */
    private void askForPairDistance(int idA, int idB) {
        EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        input.setHint("e.g. 30");

        new AlertDialog.Builder(this)
                .setTitle("Distance between anchors")
                .setMessage("Real distance in centimetres")
                .setView(input)
                .setPositiveButton("Add", (dialog, which) ->
                        addMeasuredPair(idA, idB, input.getText().toString()))
                .setNegativeButton("Cancel", (dialog, which) -> resetPendingPick())
                .setOnCancelListener(dialog -> resetPendingPick())
                .show();
    }

    /**
     * Every path through here resets the pending pick. The dialog's buttons always dismiss it, so
     * bailing out early would leave an anchor half-selected with no way to clear it.
     */
    private void addMeasuredPair(int idA, int idB, String rawDistance) {
        float centimetres;
        try {
            centimetres = Float.parseFloat(rawDistance.trim());
        } catch (NumberFormatException e) {
            centimetres = 0f;
        }

        if (centimetres <= 0f) {
            Toast.makeText(this, "Enter a distance in centimetres", Toast.LENGTH_SHORT).show();
        } else {
            groundAnchor.addPair(idA, idB, centimetres / 100f);
            Toast.makeText(this, groundAnchor.getPairs().size() + " pair(s) measured",
                    Toast.LENGTH_SHORT).show();
        }
        resetPendingPick();
    }

    private void resetPendingPick() {
        pendingPairId = -1;
        flowOverlay.setPairs(groundAnchor.getPairs(), -1);
        refreshModeUi();
    }

    /** Returning false lets the view keep its own touch handling, so the pressed tint still works. */
    @SuppressLint("ClickableViewAccessibility")
    private void bindAxisButton(int viewId, int axis) {
        findViewById(viewId).setOnTouchListener((view, event) -> {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    axisLocked[axis] = true;
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    axisLocked[axis] = false;
                    break;
                default:
                    break;
            }
            return false;
        });
    }

    /**
     * One finger slides the grid across the ground, two fingers raise or lower the ground plane.
     * The reference point is re-seeded whenever a finger lands or lifts, so changing grip does
     * not throw the grid sideways.
     */
    @SuppressLint("ClickableViewAccessibility")
    private void bindGridDragging() {
        gridView.setOnTouchListener((view, event) -> {
            if (mode == Mode.PAIRING) {
                return handlePairingTouch(event);
            }
            if (mode == Mode.SELECTING && event.getPointerCount() == 1) {
                return handleSelectionTouch(event);
            }

            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    needsTouchReseed = true;
                    touchIsTap = true;
                    touchDownX = event.getX();
                    touchDownY = event.getY();
                    return true;
                case MotionEvent.ACTION_POINTER_DOWN:
                case MotionEvent.ACTION_POINTER_UP:
                    needsTouchReseed = true;
                    touchIsTap = false;
                    return true;
                case MotionEvent.ACTION_UP:
                    if (touchIsTap) {
                        focusAt(event.getX(), event.getY());
                    }
                    touchIsTap = false;
                    return true;
                case MotionEvent.ACTION_MOVE: {
                    int pointers = event.getPointerCount();
                    if (touchIsTap && Math.hypot(event.getX() - touchDownX,
                            event.getY() - touchDownY) > touchSlop) {
                        touchIsTap = false;
                    }
                    float x = 0f;
                    float y = 0f;
                    for (int i = 0; i < pointers; i++) {
                        x += event.getX(i);
                        y += event.getY(i);
                    }
                    x /= pointers;
                    y /= pointers;

                    if (needsTouchReseed) {
                        needsTouchReseed = false;
                    } else if (pointers >= 2) {
                        gridView.adjustHeight(y - lastTouchY);
                    } else {
                        gridView.panByScreenDrag(lastTouchX, lastTouchY, x, y);
                        // Panning while locked re-registers the grid against the same anchors,
                        // so the lock survives a manual nudge instead of fighting it.
                        if (groundAnchor.isLocked()) {
                            groundAnchor.rebaseline(flowOverlay, gridView);
                        }
                    }
                    lastTouchX = x;
                    lastTouchY = y;
                    return true;
                }
                default:
                    return true;
            }
        });
    }

    private boolean handleSelectionTouch(MotionEvent event) {
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                boxStartX = event.getX();
                boxStartY = event.getY();
                selectionBox.set(boxStartX, boxStartY, boxStartX, boxStartY);
                flowOverlay.setSelectionBox(selectionBox);
                return true;
            case MotionEvent.ACTION_MOVE:
                selectionBox.set(
                        Math.min(boxStartX, event.getX()), Math.min(boxStartY, event.getY()),
                        Math.max(boxStartX, event.getX()), Math.max(boxStartY, event.getY()));
                flowOverlay.setSelectionBox(selectionBox);
                return true;
            case MotionEvent.ACTION_UP:
                stageAnchorsInBox();
                return true;
            default:
                return true;
        }
    }

    /**
     * Drives autofocus and exposure metering at the tapped point.
     *
     * The metering point comes from PreviewView's own factory, which already accounts for the
     * preview's crop, scaling and rotation, so view coordinates map to sensor coordinates without
     * any conversion here. Metering auto-cancels after a few seconds so the camera returns to
     * continuous autofocus rather than staying pinned where it was last tapped.
     */
    private void focusAt(float x, float y) {
        if (cameraControl == null) {
            return;
        }
        MeteringPoint point = previewView.getMeteringPointFactory().createPoint(x, y);
        FocusMeteringAction action = new FocusMeteringAction.Builder(point,
                FocusMeteringAction.FLAG_AF | FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(4, TimeUnit.SECONDS)
                .build();

        flowOverlay.showFocusReticle(x, y);
        ListenableFuture<FocusMeteringResult> result = cameraControl.startFocusAndMetering(action);
        result.addListener(() -> {
            boolean focused;
            try {
                focused = result.get().isFocusSuccessful();
            } catch (ExecutionException | InterruptedException | RuntimeException e) {
                focused = false;
            }
            flowOverlay.setFocusResult(focused);
        }, ContextCompat.getMainExecutor(this));
    }

    /** Taps snap to the nearest staged anchor, so an imprecise tap still picks a real point. */
    private boolean handlePairingTouch(MotionEvent event) {
        if (event.getActionMasked() != MotionEvent.ACTION_UP) {
            return true;
        }
        int picked = groundAnchor.findAnchorNear(event.getX(), event.getY(), flowOverlay);
        if (picked == -1) {
            Toast.makeText(this, "No anchor there", Toast.LENGTH_SHORT).show();
            return true;
        }

        if (pendingPairId == -1) {
            pendingPairId = picked;
            flowOverlay.setPairs(groundAnchor.getPairs(), pendingPairId);
        } else if (picked == pendingPairId) {
            resetPendingPick();
        } else {
            askForPairDistance(pendingPairId, picked);
        }
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (rotationSensor != null) {
            sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_GAME);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
        hasPreviousSensor = false;
        for (int i = 0; i < axisLocked.length; i++) {
            axisLocked[i] = false;
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);

        future.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = future.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                cameraProvider.unbindAll();
                Camera camera = flowAnalyzer == null
                        ? cameraProvider.bindToLifecycle(this,
                                CameraSelector.DEFAULT_BACK_CAMERA, preview)
                        : cameraProvider.bindToLifecycle(this,
                                CameraSelector.DEFAULT_BACK_CAMERA, preview, buildAnalysis());
                cameraControl = camera.getCameraControl();
            } catch (ExecutionException | InterruptedException e) {
                Toast.makeText(this, "Camera failed: " + e.getMessage(),
                        Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Analysis runs at a modest resolution: Lucas-Kanade gains nothing from full sensor output
     * and the per-frame cost scales with pixel count. KEEP_ONLY_LATEST means a slow frame drops
     * work rather than queueing it up behind the live preview.
     */
    private ImageAnalysis buildAnalysis() {
        ResolutionSelector resolution = new ResolutionSelector.Builder()
                .setResolutionStrategy(new ResolutionStrategy(
                        new Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build();

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        analysis.setAnalyzer(analysisExecutor, flowAnalyzer);
        return analysis;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analysisExecutor != null) {
            analysisExecutor.shutdown();
        }
        if (flowTracker != null) {
            flowTracker.release();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        SensorManager.getQuaternionFromVector(quaternion, event.values);

        if (!hasSmoothed) {
            System.arraycopy(quaternion, 0, smoothed, 0, 4);
            hasSmoothed = true;
        } else {
            // Quaternions q and -q are the same rotation; pick the nearer one before blending.
            float dot = 0f;
            for (int i = 0; i < 4; i++) {
                dot += smoothed[i] * quaternion[i];
            }
            float sign = dot < 0f ? -1f : 1f;
            for (int i = 0; i < 4; i++) {
                smoothed[i] += (quaternion[i] * sign - smoothed[i]) * SMOOTHING;
            }
            normalize(smoothed);
        }

        applyAxisLocks();

        multiply(correction, smoothed, display);
        normalize(display);
        quaternionToMatrix(display, rotationMatrix);

        if (needsRemap) {
            SensorManager.remapCoordinateSystem(rotationMatrix, remapAxisX, remapAxisY, displayMatrix);
            gridView.updateOrientation(displayMatrix);
        } else {
            gridView.updateOrientation(rotationMatrix);
        }
    }

    /**
     * Folds this frame's rotation about each held world axis into {@link #correction}, so the
     * displayed orientation stays put about that axis. The correction is never cleared, which is
     * what makes the overlay resume from where it was left rather than snapping back on release.
     */
    private void applyAxisLocks() {
        if (!hasPreviousSensor) {
            System.arraycopy(smoothed, 0, previousSensor, 0, 4);
            hasPreviousSensor = true;
            return;
        }

        boolean anyLocked = false;
        for (boolean locked : axisLocked) {
            anyLocked |= locked;
        }

        if (anyLocked) {
            // delta is this frame's rotation expressed in world coordinates.
            conjugate(previousSensor, scratchA);
            multiply(smoothed, scratchA, delta);

            for (int axis = 0; axis < 3; axis++) {
                if (!axisLocked[axis]) {
                    continue;
                }
                extractTwist(delta, axis, twist);
                conjugate(twist, scratchA);

                multiply(correction, scratchA, scratchB);
                System.arraycopy(scratchB, 0, correction, 0, 4);

                multiply(scratchA, delta, scratchB);
                System.arraycopy(scratchB, 0, delta, 0, 4);
            }
            normalize(correction);
        }

        System.arraycopy(smoothed, 0, previousSensor, 0, 4);
    }

    private String describeStatus(int trackCount) {
        if (!groundAnchor.hasAnchors()) {
            return String.format(Locale.US, "%d pts", trackCount);
        }
        String state = groundAnchor.isStalled()
                ? "STALLED" : groundAnchor.isLocked() ? "locked" : "staged";
        return String.format(Locale.US, "%d pts  %s %d/%d  %dp  h=%.2fm",
                trackCount, state,
                groundAnchor.getSurvivingCount(), groundAnchor.getAnchorCount(),
                groundAnchor.getContributingPairs(), gridView.getHeightM());
    }

    /** Drops the accumulated lock offset and any panning so the grid returns to its launch state. */
    private void resetGrid() {
        correction[0] = 1f;
        correction[1] = 0f;
        correction[2] = 0f;
        correction[3] = 0f;
        gridView.resetTransform();
        groundAnchor.release();
        flowOverlay.setSelectedIds(Collections.emptySet());
        clearPairing();
        refreshModeUi();
    }

    /** Isolates the part of {@code q} that rotates about world axis {@code axis}. */
    private static void extractTwist(float[] q, int axis, float[] out) {
        float w = q[0];
        float component = q[1 + axis];
        float norm = (float) Math.sqrt(w * w + component * component);
        out[0] = 1f;
        out[1] = 0f;
        out[2] = 0f;
        out[3] = 0f;
        if (norm < 1e-6f) {
            return;
        }
        out[0] = w / norm;
        out[1 + axis] = component / norm;
    }

    /** out = a x b, with quaternions laid out as (w, x, y, z). Must not alias out with a or b. */
    private static void multiply(float[] a, float[] b, float[] out) {
        out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
        out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    }

    private static void conjugate(float[] q, float[] out) {
        out[0] = q[0];
        out[1] = -q[1];
        out[2] = -q[2];
        out[3] = -q[3];
    }

    private static void normalize(float[] q) {
        float norm = (float) Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (norm > 1e-6f) {
            for (int i = 0; i < 4; i++) {
                q[i] /= norm;
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    /** Builds the device-to-world rotation matrix from a quaternion laid out as (w, x, y, z). */
    private static void quaternionToMatrix(float[] q, float[] m) {
        float w = q[0];
        float x = q[1];
        float y = q[2];
        float z = q[3];
        float xx = x * x;
        float yy = y * y;
        float zz = z * z;

        m[0] = 1f - 2f * (yy + zz);
        m[1] = 2f * (x * y - w * z);
        m[2] = 2f * (x * z + w * y);
        m[3] = 2f * (x * y + w * z);
        m[4] = 1f - 2f * (xx + zz);
        m[5] = 2f * (y * z - w * x);
        m[6] = 2f * (x * z - w * y);
        m[7] = 2f * (y * z + w * x);
        m[8] = 1f - 2f * (xx + yy);
    }

    /** Sensor axes are reported in the device's natural orientation, which may not be portrait. */
    private void resolveDisplayRemap() {
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        switch (rotation) {
            case Surface.ROTATION_90:
                remapAxisX = SensorManager.AXIS_Y;
                remapAxisY = SensorManager.AXIS_MINUS_X;
                break;
            case Surface.ROTATION_180:
                remapAxisX = SensorManager.AXIS_MINUS_X;
                remapAxisY = SensorManager.AXIS_MINUS_Y;
                break;
            case Surface.ROTATION_270:
                remapAxisX = SensorManager.AXIS_MINUS_Y;
                remapAxisY = SensorManager.AXIS_X;
                break;
            default:
                remapAxisX = SensorManager.AXIS_X;
                remapAxisY = SensorManager.AXIS_Y;
                break;
        }
        needsRemap = rotation != Surface.ROTATION_0;
    }

    /**
     * Field of view covered by the vertical axis of the portrait preview. The sensor's long edge
     * is normally rotated onto the screen's vertical axis, and PreviewView's default FILL_CENTER
     * scales that edge to the view height, so this angle is all the projection needs.
     */
    private float readPreviewVerticalFov() {
        try {
            CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) {
                    continue;
                }
                float[] focalLengths =
                        characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                SizeF physicalSize =
                        characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                Integer sensorOrientation =
                        characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                if (focalLengths == null || focalLengths.length == 0
                        || physicalSize == null || focalLengths[0] <= 0f) {
                    continue;
                }

                float longEdge = Math.max(physicalSize.getWidth(), physicalSize.getHeight());
                float shortEdge = Math.min(physicalSize.getWidth(), physicalSize.getHeight());
                boolean quarterTurned = sensorOrientation == null
                        || sensorOrientation == 90 || sensorOrientation == 270;
                float screenVerticalEdge = quarterTurned ? longEdge : shortEdge;

                return 2f * (float) Math.atan(screenVerticalEdge / (2f * focalLengths[0]));
            }
        } catch (CameraAccessException | IllegalArgumentException ignored) {
            // Fall through to the default below.
        }
        return (float) Math.toRadians(DEFAULT_VERTICAL_FOV_DEG);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
            @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show();
        }
    }
}
