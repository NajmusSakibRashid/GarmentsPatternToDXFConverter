package com.example.garmentpatterntodxfconverter.opticalflow;

/**
 * One feature point followed from the previous frame into the current one.
 *
 * Immutable, so a batch of these can be handed to another thread and held onto for as long as
 * the caller likes without copying or locking.
 */
public final class Track {

    /** Stable across frames for as long as the feature survives, so a point can be followed. */
    public final int id;
    public final float previousX;
    public final float previousY;
    public final float currentX;
    public final float currentY;
    /** Lucas-Kanade matching error; larger means a less confident match. */
    public final float error;

    public Track(int id, float previousX, float previousY,
            float currentX, float currentY, float error) {
        this.id = id;
        this.previousX = previousX;
        this.previousY = previousY;
        this.currentX = currentX;
        this.currentY = currentY;
        this.error = error;
    }

    public float deltaX() {
        return currentX - previousX;
    }

    public float deltaY() {
        return currentY - previousY;
    }

    /** Pixels moved since the previous frame. */
    public float displacement() {
        return (float) Math.hypot(deltaX(), deltaY());
    }

    @Override
    public String toString() {
        return "Track{id=" + id + ", x=" + currentX + ", y=" + currentY
                + ", dx=" + deltaX() + ", dy=" + deltaY() + "}";
    }
}
