// Associative memory to provide long term memory.
// Reads input address from one gathered source and scatters the result to the writable section.
// Reads input from another source as an address and stores a further source after
// threshold gating.
package s6regen;

public final class ComputeAMThresholdGatedWritable extends Compute {

    private final AMThresholdGated memory;
    private final int location;

    public ComputeAMThresholdGatedWritable(Reservoir r, int density,float threshold, int writableLocation) {
        super(r);
        memory = new AMThresholdGated(r.computeSize, density,threshold);
        location = writableLocation;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.computeBuffers[0];
        float[] workB = reservoir.computeBuffers[1];
        float[] workC = reservoir.computeBuffers[2];
        reservoir.gather(workA);
        reservoir.gather(workB);
        reservoir.gather(workC);
        memory.recallVec(workA, workA);
        reservoir.scatterWritable(workA, location);
        memory.trainVec(workC, workB);
    }

    @Override
    public int weightSize() {
        return 3 * reservoir.sizeGather();
    }

    @Override
    public int buffersRequired() {
        return 3;
    }

    @Override
    public void resetHeldState() {
        memory.reset();
    }
}
