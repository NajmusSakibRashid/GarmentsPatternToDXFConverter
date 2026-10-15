package com.example.garmentpatterntodxfconverter.opticalflow;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.util.List;

/**
 * Bridges CameraX frames into {@link OpticalFlowTracker}.
 *
 * This is the only class in the package that knows about Android. It pulls the luminance plane
 * out of each frame (already the grayscale image optical flow wants, so no colour conversion is
 * needed), turns it upright, and hands it to the tracker.
 *
 * Runs on whatever executor the analyzer was registered with, never the main thread.
 */
public class FlowAnalyzer implements ImageAnalysis.Analyzer {

    /** Called on the analysis thread once per processed frame. */
    public interface Listener {
        /**
         * @param tracks      snapshot of the current matches; safe to retain
         * @param imageWidth  width of the upright image the coordinates refer to
         * @param imageHeight height of the upright image the coordinates refer to
         */
        void onFlow(List<Track> tracks, int imageWidth, int imageHeight);
    }

    private final OpticalFlowTracker tracker;
    @Nullable
    private final Listener listener;
    private volatile boolean enabled;

    public FlowAnalyzer(OpticalFlowTracker tracker, @Nullable Listener listener) {
        this.tracker = tracker;
        this.listener = listener;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            tracker.reset();
        }
    }

    public boolean isEnabled() {
        return enabled;
    }

    public OpticalFlowTracker getTracker() {
        return tracker;
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
        Mat gray = null;
        Mat upright = null;
        try {
            if (!enabled) {
                return;
            }
            gray = toGrayscale(image);
            upright = rotateUpright(gray, image.getImageInfo().getRotationDegrees());

            tracker.processFrame(upright);

            if (listener != null) {
                listener.onFlow(tracker.getTracks(), upright.cols(), upright.rows());
            }
        } finally {
            if (upright != null && upright != gray) {
                upright.release();
            }
            if (gray != null) {
                gray.release();
            }
            image.close();
        }
    }

    /**
     * Copies plane 0 of a YUV_420_888 frame, which is the luminance channel, into a single
     * channel Mat. Rows are padded to the plane's stride, so anything wider than the image
     * has to be trimmed as it is copied.
     */
    private static Mat toGrayscale(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int width = image.getWidth();
        int height = image.getHeight();

        byte[] data = new byte[width * height];
        if (rowStride == width) {
            buffer.get(data, 0, Math.min(data.length, buffer.remaining()));
        } else {
            for (int row = 0; row < height; row++) {
                int offset = row * rowStride;
                if (offset >= buffer.limit()) {
                    break;
                }
                buffer.position(offset);
                buffer.get(data, row * width, Math.min(width, buffer.remaining()));
            }
        }

        Mat gray = new Mat(height, width, CvType.CV_8UC1);
        gray.put(0, 0, data);
        return gray;
    }

    /**
     * Rotates the sensor-orientation frame so tracked coordinates come out in the same upright
     * frame the screen uses. Returns the input untouched when no rotation is needed.
     */
    private static Mat rotateUpright(Mat source, int rotationDegrees) {
        int code;
        switch (rotationDegrees) {
            case 90:
                code = Core.ROTATE_90_CLOCKWISE;
                break;
            case 180:
                code = Core.ROTATE_180;
                break;
            case 270:
                code = Core.ROTATE_90_COUNTERCLOCKWISE;
                break;
            default:
                return source;
        }
        Mat rotated = new Mat();
        Core.rotate(source, rotated, code);
        return rotated;
    }
}
