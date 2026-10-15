package com.example.garmentpatterntodxfconverter;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import com.example.garmentpatterntodxfconverter.opticalflow.Track;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Draws optical flow tracks over the preview: a dot at each feature's current position and a
 * tail back to where it was in the previous frame.
 *
 * Purely a view onto {@link Track} data. It holds no tracker state, so swapping or extending the
 * tracking logic does not touch this class.
 */
public class FlowOverlayView extends View {

    private static final float DOT_RADIUS_DP = 3.5f;
    private static final float TRAIL_STROKE_DP = 1.8f;
    private static final int DOT_COLOR = 0xFF32E06A;
    private static final int TRAIL_COLOR = 0xFFFF7043;
    /** Tails shorter than this are noise, not motion, so they are left undrawn. */
    private static final float MIN_TRAIL_PX = 1.5f;

    private static final int ANCHOR_COLOR = 0xFFFFD54F;
    private static final int BOX_COLOR = 0xFFFFD54F;
    private static final int MARKER_COLOR = 0xFFFF4081;

    private final Paint dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint trailPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint anchorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint markerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private volatile List<Track> tracks = Collections.emptyList();
    private volatile int imageWidth;
    private volatile int imageHeight;
    private volatile Set<Integer> selectedIds = Collections.emptySet();

    @Nullable
    private RectF selectionBox;
    private List<GroundAnchor.Pair> pairs = Collections.emptyList();
    private int pendingId = -1;

    private static final int FOCUS_NONE = 0;
    private static final int FOCUS_PENDING = 1;
    private static final int FOCUS_OK = 2;
    private static final int FOCUS_FAILED = 3;

    private int focusState = FOCUS_NONE;
    private float focusX;
    private float focusY;
    private final float focusRadiusPx;
    private final Paint focusPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Runnable clearFocus = () -> {
        focusState = FOCUS_NONE;
        invalidate();
    };

    private final float dotRadiusPx;
    private final float anchorRadiusPx;
    private final float markerRadiusPx;

    public FlowOverlayView(Context context) {
        this(context, null);
    }

    public FlowOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        float density = getResources().getDisplayMetrics().density;
        dotRadiusPx = DOT_RADIUS_DP * density;
        anchorRadiusPx = DOT_RADIUS_DP * 2f * density;
        markerRadiusPx = 14f * density;

        dotPaint.setColor(DOT_COLOR);
        dotPaint.setStyle(Paint.Style.FILL);

        trailPaint.setColor(TRAIL_COLOR);
        trailPaint.setStyle(Paint.Style.STROKE);
        trailPaint.setStrokeWidth(TRAIL_STROKE_DP * density);
        trailPaint.setStrokeCap(Paint.Cap.ROUND);

        anchorPaint.setColor(ANCHOR_COLOR);
        anchorPaint.setStyle(Paint.Style.STROKE);
        anchorPaint.setStrokeWidth(2.5f * density);

        boxPaint.setColor(BOX_COLOR);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(2f * density);

        markerPaint.setColor(MARKER_COLOR);
        markerPaint.setStyle(Paint.Style.STROKE);
        markerPaint.setStrokeWidth(2.5f * density);

        focusRadiusPx = 34f * density;
        focusPaint.setStyle(Paint.Style.STROKE);
        focusPaint.setStrokeWidth(2.5f * density);
    }

    /** Shows the reticle immediately; the camera's verdict arrives later. */
    public void showFocusReticle(float x, float y) {
        focusX = x;
        focusY = y;
        focusState = FOCUS_PENDING;
        removeCallbacks(clearFocus);
        // Falls back to clearing itself in case the focus future never settles.
        postDelayed(clearFocus, 3000);
        invalidate();
    }

    public void setFocusResult(boolean focused) {
        if (focusState == FOCUS_NONE) {
            return;
        }
        focusState = focused ? FOCUS_OK : FOCUS_FAILED;
        removeCallbacks(clearFocus);
        postDelayed(clearFocus, 800);
        invalidate();
    }

    /** Safe to call from the analysis thread. */
    public void setTracks(List<Track> tracks, int imageWidth, int imageHeight) {
        this.tracks = tracks == null ? Collections.<Track>emptyList() : tracks;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        postInvalidate();
    }

    public void clear() {
        setTracks(Collections.<Track>emptyList(), imageWidth, imageHeight);
    }

    /** Staged anchors, drawn ringed so they stand out from ordinary tracks. */
    public void setSelectedIds(Set<Integer> ids) {
        this.selectedIds = ids == null ? Collections.<Integer>emptySet() : ids;
        postInvalidate();
    }

    public void setSelectionBox(@Nullable RectF box) {
        this.selectionBox = box;
        invalidate();
    }

    /** Measured pairs, drawn as links between their two anchors. */
    public void setPairs(List<GroundAnchor.Pair> pairs, int pendingId) {
        this.pairs = pairs == null ? Collections.<GroundAnchor.Pair>emptyList() : pairs;
        this.pendingId = pendingId;
        invalidate();
    }

    /**
     * Maps a point in analysis-image pixels to view pixels, matching PreviewView's default
     * FILL_CENTER: scale to cover, then centre the overflow.
     *
     * The single source of truth for this mapping — {@code onDraw} and anchor ray-casting both go
     * through it, so the two can never drift apart.
     *
     * @return false if no frame has been seen yet, leaving {@code out} untouched
     */
    public boolean imageToView(float imageX, float imageY, float[] out) {
        int sourceWidth = imageWidth;
        int sourceHeight = imageHeight;
        if (sourceWidth == 0 || sourceHeight == 0) {
            return false;
        }
        float scale = Math.max((float) getWidth() / sourceWidth, (float) getHeight() / sourceHeight);
        out[0] = imageX * scale + (getWidth() - sourceWidth * scale) * 0.5f;
        out[1] = imageY * scale + (getHeight() - sourceHeight * scale) * 0.5f;
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        drawTracks(canvas);

        RectF box = selectionBox;
        if (box != null) {
            canvas.drawRect(box, boxPaint);
        }

        drawPairs(canvas);
        drawFocusReticle(canvas);
    }

    private void drawFocusReticle(Canvas canvas) {
        if (focusState == FOCUS_NONE) {
            return;
        }
        focusPaint.setColor(focusState == FOCUS_OK ? DOT_COLOR
                : focusState == FOCUS_FAILED ? 0xFFFFB300 : 0xFFFFFFFF);
        canvas.drawCircle(focusX, focusY, focusRadiusPx, focusPaint);
        canvas.drawCircle(focusX, focusY, focusRadiusPx * 0.12f, focusPaint);
    }

    private void drawPairs(Canvas canvas) {
        float[] a = new float[2];
        float[] b = new float[2];
        for (GroundAnchor.Pair pair : pairs) {
            if (locate(pair.idA, a) && locate(pair.idB, b)) {
                canvas.drawLine(a[0], a[1], b[0], b[1], markerPaint);
            }
        }
        if (pendingId != -1 && locate(pendingId, a)) {
            canvas.drawCircle(a[0], a[1], markerRadiusPx, markerPaint);
        }
    }

    /** Resolves a track id to its current view position. */
    private boolean locate(int id, float[] out) {
        for (Track track : tracks) {
            if (track.id == id) {
                return imageToView(track.currentX, track.currentY, out);
            }
        }
        return false;
    }

    private void drawTracks(Canvas canvas) {
        List<Track> snapshot = tracks;
        Set<Integer> selected = selectedIds;
        if (snapshot.isEmpty() || imageWidth == 0 || imageHeight == 0) {
            return;
        }

        float[] point = new float[2];
        for (Track track : snapshot) {
            if (!imageToView(track.currentX, track.currentY, point)) {
                return;
            }
            float x = point[0];
            float y = point[1];
            imageToView(track.previousX, track.previousY, point);

            if (Math.hypot(x - point[0], y - point[1]) >= MIN_TRAIL_PX) {
                canvas.drawLine(point[0], point[1], x, y, trailPaint);
            }
            canvas.drawCircle(x, y, dotRadiusPx, dotPaint);
            if (selected.contains(track.id)) {
                canvas.drawCircle(x, y, anchorRadiusPx, anchorPaint);
            }
        }
    }
}
