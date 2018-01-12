// Sends some random numbers to the writable section
package s6regen;

public final class ComputeRandomWritable extends Compute {

    private final RNG rng;
    private final int location;

    public ComputeRandomWritable(Reservoir r,int writableLocation) {
        super(r);
        rng=new RNG();
        location = writableLocation;
    }

    @Override
    public void compute() {
        float[] workA = reservoir.computeBuffers[0];
        for(int i=0;i<reservoir.computeSize;i++){
            workA[i]=rng.nextFloatSym();
        }
        reservoir.scatterWritable(workA, location);
    }

    @Override
    public int weightSize() {
        return 0;
    }

    @Override
    public int buffersRequired() {
        return 1;
    }

}
