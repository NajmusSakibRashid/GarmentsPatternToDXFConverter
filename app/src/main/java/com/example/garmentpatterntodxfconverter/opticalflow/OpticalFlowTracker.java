package com.example.garmentpatterntodxfconverter.opticalflow;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Shi-Tomasi corner detection plus pyramidal Lucas-Kanade optical flow.
 *
 * Deliberately free of any Android dependency: it takes a grayscale {@link Mat} in and hands
 * {@link Track} objects out. Nothing here draws, and nothing here knows about the camera or the
 * screen, so the results can be fed straight into whatever analysis comes next.
 *
 * Call {@link #processFrame} from one thread only. The getters are safe to call from any thread
 * and always return a complete, self-consistent snapshot.
 */
public class OpticalFlowTracker {

    // --- Shi-Tomasi corner detection ---
    private int maxCorners = 200;
    private double qualityLevel = 0.3;
    private double minDistance = 7.0;
    private int blockSize = 7;

    // --- Lucas-Kanade ---
    private Size windowSize = new Size(21, 21);
    private int maxPyramidLevel = 3;
    private TermCriteria criteria =
            new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 10, 0.03);

    /** Top the point set back up once survivors fall below this. */
    private int reseedThreshold = 120;
    /** Matches worse than this are dropped. Set to 0 to keep everything. */
    private float maxError = 0f;

    private final Mat previousGray = new Mat();
    private final MatOfPoint2f previousPoints = new MatOfPoint2f();
    private final MatOfPoint2f nextPoints = new MatOfPoint2f();
    private final MatOfByte status = new MatOfByte();
    private final MatOfFloat error = new MatOfFloat();

    private int[] trackIds = new int[0];
    private int nextTrackId = 1;

    private volatile List<Track> tracks = Collections.emptyList();
    private volatile long frameIndex;
    private volatile int frameWidth;
    private volatile int frameHeight;

    // ---------------------------------------------------------------- inputs

    /**
     * Advances the tracker by one frame.
     *
     * @param grayFrame single channel 8-bit image; not retained, the caller may release it
     */
    public void processFrame(Mat grayFrame) {
        if (grayFrame == null || grayFrame.empty()) {
            return;
        }
        frameWidth = grayFrame.cols();
        frameHeight = grayFrame.rows();

        // First frame, or everything was lost: there is nothing to match against yet.
        if (previousGray.empty() || previousPoints.rows() == 0) {
            detectFeatures(grayFrame, false);
            grayFrame.copyTo(previousGray);
            tracks = Collections.emptyList();
            frameIndex++;
            return;
        }

        Video.calcOpticalFlowPyrLK(previousGray, grayFrame, previousPoints, nextPoints,
                status, error, windowSize, maxPyramidLevel, criteria);

        byte[] statusArray = status.toArray();
        Point[] before = previousPoints.toArray();
        Point[] after = nextPoints.toArray();
        float[] errors = error.toArray();

        int capacity = Math.min(statusArray.length, Math.min(before.length, after.length));
        List<Track> found = new ArrayList<>(capacity);
        List<Point> survivors = new ArrayList<>(capacity);
        int[] survivingIds = new int[capacity];
        int kept = 0;

        for (int i = 0; i < capacity; i++) {
            if (statusArray[i] != 1) {
                continue;
            }
            float matchError = i < errors.length ? errors[i] : 0f;
            if (maxError > 0f && matchError > maxError) {
                continue;
            }
            int id = i < trackIds.length ? trackIds[i] : 0;
            found.add(new Track(id,
                    (float) before[i].x, (float) before[i].y,
                    (float) after[i].x, (float) after[i].y,
                    matchError));
            survivors.add(after[i]);
            survivingIds[kept++] = id;
        }

        tracks = Collections.unmodifiableList(found);

        if (survivors.isEmpty()) {
            previousPoints.release();
            trackIds = new int[0];
        } else {
            previousPoints.fromList(survivors);
            trackIds = Arrays.copyOf(survivingIds, kept);
        }

        grayFrame.copyTo(previousGray);

        if (kept < reseedThreshold) {
            detectFeatures(grayFrame, true);
        }

        frameIndex++;
    }

    /**
     * Finds fresh corners. When {@code keepExisting} is set, the surviving points are masked out
     * first so new corners fill the gaps instead of piling up on features already being followed.
     */
    private void detectFeatures(Mat grayFrame, boolean keepExisting) {
        int existing = keepExisting ? previousPoints.rows() : 0;
        int wanted = maxCorners - existing;
        if (wanted <= 0) {
            return;
        }

        Mat mask = new Mat(grayFrame.size(), CvType.CV_8UC1, new Scalar(255));
        Point[] existingPoints = existing > 0 ? previousPoints.toArray() : new Point[0];
        for (Point point : existingPoints) {
            Imgproc.circle(mask, point, (int) Math.max(1, minDistance), new Scalar(0), -1);
        }

        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(grayFrame, corners, wanted, qualityLevel, minDistance,
                mask, blockSize, false, 0.04);

        List<Point> merged = new ArrayList<>(Arrays.asList(existingPoints));
        List<Point> fresh = corners.toList();
        merged.addAll(fresh);

        int[] mergedIds = new int[merged.size()];
        for (int i = 0; i < existingPoints.length && i < trackIds.length; i++) {
            mergedIds[i] = trackIds[i];
        }
        for (int i = existingPoints.length; i < mergedIds.length; i++) {
            mergedIds[i] = nextTrackId++;
        }

        if (!merged.isEmpty()) {
            previousPoints.fromList(merged);
            trackIds = mergedIds;
        }

        mask.release();
        corners.release();
    }

    /** Forgets every point and starts detection over on the next frame. */
    public void reset() {
        previousGray.release();
        previousPoints.release();
        trackIds = new int[0];
        tracks = Collections.emptyList();
        frameIndex = 0;
    }

    /** Frees native memory. The tracker is reusable afterwards. */
    public void release() {
        reset();
        nextPoints.release();
        status.release();
        error.release();
    }

    // --------------------------------------------------------------- outputs

    /** Points matched between the previous frame and the current one. Never null. */
    public List<Track> getTracks() {
        return tracks;
    }

    public int getTrackCount() {
        return tracks.size();
    }

    /** Counts every frame fed in, including ones that only seeded features. */
    public long getFrameIndex() {
        return frameIndex;
    }

    public int getFrameWidth() {
        return frameWidth;
    }

    public int getFrameHeight() {
        return frameHeight;
    }

    /** Mean pixel displacement across all live tracks; 0 when nothing is being followed. */
    public float getMeanDisplacement() {
        List<Track> snapshot = tracks;
        if (snapshot.isEmpty()) {
            return 0f;
        }
        float total = 0f;
        for (Track track : snapshot) {
            total += track.displacement();
        }
        return total / snapshot.size();
    }

    // -------------------------------------------------------------- tunables

    public void setMaxCorners(int maxCorners) {
        this.maxCorners = maxCorners;
    }

    public void setQualityLevel(double qualityLevel) {
        this.qualityLevel = qualityLevel;
    }

    public void setMinDistance(double minDistance) {
        this.minDistance = minDistance;
    }

    public void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }

    public void setWindowSize(int side) {
        this.windowSize = new Size(side, side);
    }

    public void setMaxPyramidLevel(int maxPyramidLevel) {
        this.maxPyramidLevel = maxPyramidLevel;
    }

    public void setTermCriteria(TermCriteria criteria) {
        this.criteria = criteria;
    }

    public void setReseedThreshold(int reseedThreshold) {
        this.reseedThreshold = reseedThreshold;
    }

    public void setMaxError(float maxError) {
        this.maxError = maxError;
    }

    public int getMaxCorners() {
        return maxCorners;
    }
}
