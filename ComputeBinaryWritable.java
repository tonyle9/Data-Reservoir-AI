// Get a binary snapshot of the reservoir with bias 
package s6regen;

public final class ComputeBinaryWritable extends Compute {

    private final int location;

    public ComputeBinaryWritable(Reservoir r,int writableLocation) {
        super(r);
        location = writableLocation;
    }

    @Override
    public void compute() {
        float[] wt=reservoir.weights;
        float[] workA = reservoir.computeBuffers[0];
        reservoir.gather(workA);
        int wtIdx=reservoir.weightIndex;
        for(int i=0;i<reservoir.computeSize;i++){
            workA[i]=(workA[i]+wt[wtIdx++])>0 ?1f:-1f;
        }
        reservoir.weightIndex=wtIdx;
        reservoir.scatterWritable(workA, location);
    }

    @Override
    public int weightSize() {
        return reservoir.sizeGather()+reservoir.computeSize;
    }

    @Override
    public int buffersRequired() {
        return 1;
    }
}
