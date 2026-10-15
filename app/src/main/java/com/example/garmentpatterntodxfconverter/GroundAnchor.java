package com.example.garmentpatterntodxfconverter;

import com.example.garmentpatterntodxfconverter.opticalflow.Track;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Holds the grid onto a set of tracked feature points: solves the plane's height from pairs of
 * anchors whose real separation is known, and drives the grid's translation from how those
 * anchors move.
 *
 * <h3>Height</h3>
 * Casting a point that really lies at height {@code h'} onto the assumed plane {@code h} stretches
 * its position by exactly {@code h/h'}, so a pair separated by a known {@code D} measures as
 * {@code S = (h/h')·D} and yields {@code h' = h·D/S}. {@code S} is linear in {@code h}, so this
 * lands on the answer in one step rather than converging — re-running it every frame is tracking,
 * not refinement, and keeps the height honest as orientation drifts.
 *
 * <h3>Translation</h3>
 * The device has no positional tracking, so the world frame is pinned to the phone and a static
 * feature drifts through it exactly as far as the phone physically moved. Holding the grid at a
 * fixed offset from those features is what glues it to the real surface. Rotation needs no special
 * handling: the ray-cast already folds in the live IMU orientation.
 *
 * Call everything on the main thread — the ray-cast reads view state the main thread writes.
 */
public class GroundAnchor {

    /** Below this, averaging no longer suppresses tracking noise, so translation stops. */
    private static final int MIN_SURVIVING = 3;
    /** Per-frame weight of a new height estimate; lower is steadier but laggier. */
    private static final float HEIGHT_SMOOTHING = 0.15f;
    /** Taps further than this from an anchor do not select it. */
    private static final float PICK_RADIUS_PX = 90f;

    /** Two anchors whose real-world separation the user has measured. */
    public static final class Pair {
        public final int idA;
        public final int idB;
        public final float distanceM;

        Pair(int idA, int idB, float distanceM) {
            this.idA = idA;
            this.idB = idB;
            this.distanceM = distanceM;
        }
    }

    /** Track id to that anchor's fixed coordinate in the grid's own frame. */
    private final Map<Integer, float[]> anchors = new HashMap<>();
    private final List<Pair> pairs = new ArrayList<>();

    private List<Track> lastTracks = Collections.emptyList();
    private final Map<Integer, Track> liveById = new HashMap<>();

    private boolean locked;
    private int survivingCount;
    private float solvedHeightM;
    private int contributingPairs;

    private final float[] viewPoint = new float[2];
    private final float[] groundPoint = new float[2];
    private final float[] groundB = new float[2];

    // ------------------------------------------------------------- selection

    /**
     * Records the chosen tracks as anchors without moving the grid. Selection, measurement and
     * locking are separate steps because the anchors lie on a plane whose height is not yet known.
     */
    public void stage(List<Track> tracks, Set<Integer> ids,
            FlowOverlayView mapper, GroundGridView grid) {
        anchors.clear();
        pairs.clear();
        solvedHeightM = 0f;
        setTracks(tracks);
        for (Track track : lastTracks) {
            if (ids.contains(track.id) && resolveGround(track, mapper, grid)) {
                anchors.put(track.id, new float[]{
                        groundPoint[0] - grid.getOffsetX(),
                        groundPoint[1] - grid.getOffsetY()});
            }
        }
        survivingCount = anchors.size();
    }

    /** @return the id of the nearest live anchor to a view-space point, or -1 if none is close */
    public int findAnchorNear(float viewX, float viewY, FlowOverlayView mapper) {
        int best = -1;
        float bestDistance = PICK_RADIUS_PX;
        for (Track track : lastTracks) {
            if (!anchors.containsKey(track.id)
                    || !mapper.imageToView(track.currentX, track.currentY, viewPoint)) {
                continue;
            }
            float distance = (float) Math.hypot(viewPoint[0] - viewX, viewPoint[1] - viewY);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = track.id;
            }
        }
        return best;
    }

    public void addPair(int idA, int idB, float distanceM) {
        if (idA != idB && distanceM > 0f
                && anchors.containsKey(idA) && anchors.containsKey(idB)) {
            pairs.add(new Pair(idA, idB, distanceM));
        }
    }

    public List<Pair> getPairs() {
        return Collections.unmodifiableList(pairs);
    }

    public boolean hasPairs() {
        return !pairs.isEmpty();
    }

    // ---------------------------------------------------------------- update

    /**
     * Re-solves the height from every measurable pair, then, while locked, moves the grid so the
     * anchors stay over the same physical features.
     */
    public void update(List<Track> tracks, FlowOverlayView mapper, GroundGridView grid) {
        setTracks(tracks);
        solveHeight(mapper, grid);

        if (!locked || anchors.isEmpty()) {
            return;
        }

        List<Float> candidateX = new ArrayList<>(anchors.size());
        List<Float> candidateY = new ArrayList<>(anchors.size());

        for (Track track : lastTracks) {
            float[] anchor = anchors.get(track.id);
            // A missed ray means the point swung toward the horizon this frame, not that the
            // anchor died, so it is skipped rather than dropped.
            if (anchor == null || !resolveGround(track, mapper, grid)) {
                continue;
            }
            candidateX.add(groundPoint[0] - anchor[0]);
            candidateY.add(groundPoint[1] - anchor[1]);
        }

        survivingCount = candidateX.size();
        if (survivingCount >= MIN_SURVIVING) {
            grid.setPan(median(candidateX), median(candidateY));
        }
    }

    /**
     * Averages {@code h·D/S} over the pairs that still resolve, smooths the result, and rescales
     * the stored anchors and the grid's pan to match.
     *
     * The rescale is essential: ray-cast positions are proportional to the plane height, so moving
     * the plane without scaling the recorded coordinates by the same factor would register as a
     * sudden translation and throw the grid sideways.
     */
    private void solveHeight(FlowOverlayView mapper, GroundGridView grid) {
        contributingPairs = 0;
        if (pairs.isEmpty()) {
            return;
        }

        float currentHeight = grid.getHeightM();
        List<Float> estimates = new ArrayList<>(pairs.size());

        for (Pair pair : pairs) {
            Track a = liveById.get(pair.idA);
            Track b = liveById.get(pair.idB);
            if (a == null || b == null
                    || !resolveGround(a, mapper, grid)) {
                continue;
            }
            groundB[0] = groundPoint[0];
            groundB[1] = groundPoint[1];
            if (!resolveGround(b, mapper, grid)) {
                continue;
            }
            float separation = (float) Math.hypot(
                    groundPoint[0] - groundB[0], groundPoint[1] - groundB[1]);
            if (separation < 1e-4f) {
                continue;
            }
            estimates.add(currentHeight * pair.distanceM / separation);
        }

        contributingPairs = estimates.size();
        if (contributingPairs == 0) {
            return;
        }

        float target = median(estimates);
        solvedHeightM = solvedHeightM <= 0f
                ? target
                : solvedHeightM + (target - solvedHeightM) * HEIGHT_SMOOTHING;

        float applied = grid.setHeightM(solvedHeightM);
        solvedHeightM = applied;

        float scale = applied / currentHeight;
        if (Math.abs(scale - 1f) > 1e-6f) {
            rescale(scale, grid);
        }
    }

    /** Keeps anchors and pan consistent with a plane that just moved. */
    private void rescale(float scale, GroundGridView grid) {
        for (float[] anchor : anchors.values()) {
            anchor[0] *= scale;
            anchor[1] *= scale;
        }
        grid.setPan(grid.getOffsetX() * scale, grid.getOffsetY() * scale);
    }

    /** Re-derives every anchor coordinate against the grid's current offset. */
    public void rebaseline(FlowOverlayView mapper, GroundGridView grid) {
        if (anchors.isEmpty()) {
            return;
        }
        for (Track track : lastTracks) {
            float[] anchor = anchors.get(track.id);
            if (anchor != null && resolveGround(track, mapper, grid)) {
                anchor[0] = groundPoint[0] - grid.getOffsetX();
                anchor[1] = groundPoint[1] - grid.getOffsetY();
            }
        }
    }

    private void setTracks(List<Track> tracks) {
        lastTracks = tracks == null ? Collections.<Track>emptyList() : tracks;
        liveById.clear();
        for (Track track : lastTracks) {
            liveById.put(track.id, track);
        }
    }

    private boolean resolveGround(Track track, FlowOverlayView mapper, GroundGridView grid) {
        return mapper.imageToView(track.currentX, track.currentY, viewPoint)
                && grid.screenToGround(viewPoint[0], viewPoint[1], groundPoint);
    }

    /** Median, not mean: one stray point or mistyped distance would drag a mean badly. */
    private static float median(List<Float> values) {
        Float[] sorted = values.toArray(new Float[0]);
        Arrays.sort(sorted);
        int middle = sorted.length / 2;
        if (sorted.length % 2 == 1) {
            return sorted[middle];
        }
        return (sorted[middle - 1] + sorted[middle]) * 0.5f;
    }

    // ----------------------------------------------------------------- state

    public void setLocked(boolean locked) {
        this.locked = locked;
    }

    public boolean isLocked() {
        return locked;
    }

    public void release() {
        anchors.clear();
        pairs.clear();
        locked = false;
        survivingCount = 0;
        solvedHeightM = 0f;
        contributingPairs = 0;
    }

    public boolean hasAnchors() {
        return !anchors.isEmpty();
    }

    public int getAnchorCount() {
        return anchors.size();
    }

    /** Anchors that resolved on the most recent update. */
    public int getSurvivingCount() {
        return survivingCount;
    }

    /** Pairs that still had both ends alive on the most recent solve. */
    public int getContributingPairs() {
        return contributingPairs;
    }

    /** True when locked but no longer able to translate, because too few anchors survive. */
    public boolean isStalled() {
        return locked && !anchors.isEmpty() && survivingCount < MIN_SURVIVING;
    }

    public Set<Integer> getAnchorIds() {
        return new HashSet<>(anchors.keySet());
    }
}
