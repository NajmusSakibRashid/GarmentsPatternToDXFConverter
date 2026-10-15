package com.example.garmentpatterntodxfconverter;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

/**
 * Draws a world-locked horizontal grid ("the ground") over the camera preview.
 *
 * World frame is Android's ENU convention: X east, Y north, Z up, origin at the device.
 * The ground is the plane Z = -heightM. Grid lines are projected through a pinhole model whose
 * focal length is derived from the real camera's field of view, so the grid stays registered
 * with the scene as the phone moves.
 *
 * Two levels are drawn: thick 1 m majors, and each major cell subdivided 10x10 by thin
 * 10 cm minors. The minors are held to a short radius because beyond a few metres their
 * spacing falls below a pixel and they alias into a solid wash.
 */
public class GroundGridView extends View {

    private static final float DEFAULT_HEIGHT_M = 1.4f;
    private static final float MIN_HEIGHT_M = 0.1f;
    private static final float MAX_HEIGHT_M = 20f;
    /** Height change produced by dragging two fingers the full height of the view. */
    private static final float HEIGHT_DRAG_RANGE_M = 3f;
    /** Drags landing further away than this are ignored; near the horizon they explode. */
    private static final float MAX_PICK_M = 60f;

    private static final float MAJOR_SPACING_M = 1f;
    private static final int MAJOR_HALF_LINES = 25;
    private static final float MAJOR_HALF_LENGTH_M = 25f;
    private static final float MAJOR_FADE_START_M = 3f;
    private static final float MAJOR_FADE_END_M = 25f;
    private static final float MAJOR_STROKE_DP = 2.4f;
    private static final int MAJOR_ALPHA = 235;
    private static final int MAJOR_COLOR = 0xFF2D9BFF;

    /** Subdivisions of each major cell, per axis. */
    private static final int MINOR_PER_MAJOR = 10;
    private static final float MINOR_SPACING_M = MAJOR_SPACING_M / MINOR_PER_MAJOR;
    private static final int MINOR_HALF_LINES = 40;
    private static final float MINOR_HALF_LENGTH_M = 4f;
    private static final float MINOR_FADE_START_M = 1.5f;
    private static final float MINOR_FADE_END_M = 4f;
    private static final float MINOR_STROKE_DP = 0.85f;
    private static final int MINOR_ALPHA = 120;
    private static final int MINOR_COLOR = 0xFF8FCBFF;

    /** Lines are subdivided so each piece can be culled and faded independently. */
    private static final int SEGMENTS = 16;
    private static final float NEAR_M = 0.2f;
    private static final int BANDS = 8;
    private static final float OFFSCREEN_MARGIN_PX = 64f;

    private static final int MINOR_LINE_COUNT =
            2 * (2 * MINOR_HALF_LINES + 1 - (2 * MINOR_HALF_LINES / MINOR_PER_MAJOR + 1));
    private static final int MAJOR_LINE_COUNT = 2 * (2 * MAJOR_HALF_LINES + 1);
    private static final int BAND_CAPACITY =
            Math.max(MINOR_LINE_COUNT, MAJOR_LINE_COUNT) * SEGMENTS * 4;

    private float verticalFovRad = (float) Math.toRadians(65);

    /** Device-to-world rotation, row major. */
    private final float[] rot = new float[9];
    private boolean hasRotation;

    private float heightM = DEFAULT_HEIGHT_M;
    private float offsetX;
    private float offsetY;

    private final Paint majorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint minorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final float[][] bandPoints = new float[BANDS][BAND_CAPACITY];
    private final int[] bandLength = new int[BANDS];

    private final float[] camX = new float[SEGMENTS + 1];
    private final float[] camY = new float[SEGMENTS + 1];
    private final float[] camZ = new float[SEGMENTS + 1];
    private final float[] groundDist = new float[SEGMENTS + 1];

    private final float[] pickFrom = new float[2];
    private final float[] pickTo = new float[2];

    private float focalPx;
    private float originX;
    private float originY;
    private float fadeStartM;
    private float fadeEndM;

    public GroundGridView(Context context) {
        this(context, null);
    }

    public GroundGridView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        float density = getResources().getDisplayMetrics().density;

        majorPaint.setColor(MAJOR_COLOR);
        majorPaint.setStyle(Paint.Style.STROKE);
        majorPaint.setStrokeWidth(MAJOR_STROKE_DP * density);

        minorPaint.setColor(MINOR_COLOR);
        minorPaint.setStyle(Paint.Style.STROKE);
        minorPaint.setStrokeWidth(MINOR_STROKE_DP * density);
    }

    /** Vertical field of view of the displayed preview, in radians. */
    public void setVerticalFov(float radians) {
        if (radians > 0f) {
            verticalFovRad = radians;
        }
    }

    /** Device-to-world rotation matrix, already remapped to the display orientation. */
    public void updateOrientation(float[] rotationMatrix) {
        System.arraycopy(rotationMatrix, 0, rot, 0, 9);
        hasRotation = true;
        invalidate();
    }

    /**
     * Slides the grid so the ground point under {@code from} ends up under {@code to}, which
     * makes the grid track the finger exactly rather than at some arbitrary gain.
     */
    public void panByScreenDrag(float fromX, float fromY, float toX, float toY) {
        if (!screenToGround(fromX, fromY, pickFrom) || !screenToGround(toX, toY, pickTo)) {
            return;
        }
        offsetX += pickTo[0] - pickFrom[0];
        offsetY += pickTo[1] - pickFrom[1];
        invalidate();
    }

    /** Raises the ground plane toward the camera as the drag moves up the screen. */
    public void adjustHeight(float dragPixels) {
        int height = getHeight();
        if (height == 0) {
            return;
        }
        heightM += dragPixels / height * HEIGHT_DRAG_RANGE_M;
        heightM = Math.max(MIN_HEIGHT_M, Math.min(MAX_HEIGHT_M, heightM));
        invalidate();
    }

    public void resetTransform() {
        offsetX = 0f;
        offsetY = 0f;
        heightM = DEFAULT_HEIGHT_M;
        invalidate();
    }

    public void setPan(float x, float y) {
        offsetX = x;
        offsetY = y;
        invalidate();
    }

    public float getOffsetX() {
        return offsetX;
    }

    public float getOffsetY() {
        return offsetY;
    }

    public float getHeightM() {
        return heightM;
    }

    /** Clamped to the usable range; returns the value actually stored. */
    public float setHeightM(float metres) {
        heightM = Math.max(MIN_HEIGHT_M, Math.min(MAX_HEIGHT_M, metres));
        invalidate();
        return heightM;
    }

    /** Casts the screen point onto the ground plane. Returns false if the ray misses. */
    public boolean screenToGround(float screenX, float screenY, float[] out) {
        if (!hasRotation || !ensureProjection()) {
            return false;
        }

        // Undo the projection to get a ray in camera space, then walk it back to world space.
        float deviceX = (screenX - originX) / focalPx;
        float deviceY = -(screenY - originY) / focalPx;
        float deviceZ = -1f;

        float worldX = rot[0] * deviceX + rot[1] * deviceY + rot[2] * deviceZ;
        float worldY = rot[3] * deviceX + rot[4] * deviceY + rot[5] * deviceZ;
        float worldZ = rot[6] * deviceX + rot[7] * deviceY + rot[8] * deviceZ;

        if (worldZ > -1e-4f) {
            return false;
        }
        float t = -heightM / worldZ;
        if (t <= 0f || t > MAX_PICK_M) {
            return false;
        }

        out[0] = t * worldX;
        out[1] = t * worldY;
        return true;
    }

    private boolean ensureProjection() {
        int width = getWidth();
        int height = getHeight();
        if (width == 0 || height == 0) {
            return false;
        }
        focalPx = (height * 0.5f) / (float) Math.tan(verticalFovRad * 0.5f);
        originX = width * 0.5f;
        originY = height * 0.5f;
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (!hasRotation || !ensureProjection()) {
            return;
        }
        int width = getWidth();
        int height = getHeight();

        drawGrid(canvas, minorPaint, MINOR_SPACING_M, MINOR_HALF_LINES, MINOR_HALF_LENGTH_M,
                MINOR_FADE_START_M, MINOR_FADE_END_M, MINOR_ALPHA, true, width, height);
        drawGrid(canvas, majorPaint, MAJOR_SPACING_M, MAJOR_HALF_LINES, MAJOR_HALF_LENGTH_M,
                MAJOR_FADE_START_M, MAJOR_FADE_END_M, MAJOR_ALPHA, false, width, height);
    }

    /**
     * The drawn patch always stays centred on the viewer and translation only shifts the line
     * phase, so the grid still reads as infinite no matter how far it has been panned. Reducing
     * the phase modulo the major spacing keeps minors aligned under majors.
     *
     * @param skipMajors drop every Nth line so the minor pass does not redraw under the majors
     */
    private void drawGrid(Canvas canvas, Paint paint, float spacing, int halfLines,
            float halfLength, float fadeStart, float fadeEnd, int maxAlpha,
            boolean skipMajors, int width, int height) {
        fadeStartM = fadeStart;
        fadeEndM = fadeEnd;

        float phaseX = offsetX - (float) Math.floor(offsetX / MAJOR_SPACING_M) * MAJOR_SPACING_M;
        float phaseY = offsetY - (float) Math.floor(offsetY / MAJOR_SPACING_M) * MAJOR_SPACING_M;

        for (int i = 0; i < BANDS; i++) {
            bandLength[i] = 0;
        }

        for (int axis = 0; axis < 2; axis++) {
            float phase = axis == 0 ? phaseX : phaseY;
            for (int i = -halfLines; i <= halfLines; i++) {
                if (skipMajors && i % MINOR_PER_MAJOR == 0) {
                    continue;
                }
                projectLine(axis, i * spacing + phase, halfLength);
                for (int s = 0; s < SEGMENTS; s++) {
                    emitSegment(s, s + 1, width, height);
                }
            }
        }

        for (int b = 0; b < BANDS; b++) {
            if (bandLength[b] == 0) {
                continue;
            }
            paint.setAlpha(Math.round(maxAlpha * (b + 1f) / BANDS));
            canvas.drawLines(bandPoints[b], 0, bandLength[b], paint);
        }
    }

    /** Samples one grid line into camera space. axis 0 runs north, axis 1 runs east. */
    private void projectLine(int axis, float offset, float halfLength) {
        float worldZ = -heightM;
        for (int s = 0; s <= SEGMENTS; s++) {
            float along = -halfLength + (2f * halfLength) * s / SEGMENTS;
            float worldX = axis == 0 ? offset : along;
            float worldY = axis == 0 ? along : offset;

            // Device space is the transpose of the device-to-world rotation.
            float dx = rot[0] * worldX + rot[3] * worldY + rot[6] * worldZ;
            float dy = rot[1] * worldX + rot[4] * worldY + rot[7] * worldZ;
            float dz = rot[2] * worldX + rot[5] * worldY + rot[8] * worldZ;

            // The back camera looks along -Z of the device.
            camX[s] = dx;
            camY[s] = dy;
            camZ[s] = -dz;
            groundDist[s] = (float) Math.hypot(worldX, worldY);
        }
    }

    private void emitSegment(int a, int b, int width, int height) {
        float ax = camX[a];
        float ay = camY[a];
        float az = camZ[a];
        float bx = camX[b];
        float by = camY[b];
        float bz = camZ[b];

        if (az < NEAR_M && bz < NEAR_M) {
            return;
        }

        float distance = (groundDist[a] + groundDist[b]) * 0.5f;
        float fade = 1f - (distance - fadeStartM) / (fadeEndM - fadeStartM);
        if (fade <= 0f) {
            return;
        }
        if (fade > 1f) {
            fade = 1f;
        }

        if (az < NEAR_M) {
            float t = (NEAR_M - az) / (bz - az);
            ax += (bx - ax) * t;
            ay += (by - ay) * t;
            az = NEAR_M;
        } else if (bz < NEAR_M) {
            float t = (NEAR_M - bz) / (az - bz);
            bx += (ax - bx) * t;
            by += (ay - by) * t;
            bz = NEAR_M;
        }

        float u0 = originX + focalPx * ax / az;
        float v0 = originY - focalPx * ay / az;
        float u1 = originX + focalPx * bx / bz;
        float v1 = originY - focalPx * by / bz;

        float m = OFFSCREEN_MARGIN_PX;
        boolean offscreen = (u0 < -m && u1 < -m)
                || (u0 > width + m && u1 > width + m)
                || (v0 < -m && v1 < -m)
                || (v0 > height + m && v1 > height + m);
        if (offscreen) {
            return;
        }

        int band = (int) (fade * BANDS);
        if (band >= BANDS) {
            band = BANDS - 1;
        }

        int n = bandLength[band];
        if (n + 4 > BAND_CAPACITY) {
            return;
        }
        float[] buffer = bandPoints[band];
        buffer[n] = u0;
        buffer[n + 1] = v0;
        buffer[n + 2] = u1;
        buffer[n + 3] = v1;
        bandLength[band] = n + 4;
    }
}
