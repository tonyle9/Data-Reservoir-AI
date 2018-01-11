// Associative memory to provide long term memory.
// Reads input address from one gathered source and scatters the result.
// Reads input from another source as an address and stores to the writable area.
package s6regen;

public final class ComputeAMWritable extends Compute{
    
    private final AM memory;
    private final int location;
    
    ComputeAMWritable(Reservoir r,int density,int writableLocation){
        super(r);
        memory=new AM(r.computeSize,density);
        location=writableLocation;
    }
    
    @Override
    public void compute() {
       float[] workA=reservoir.computeBuffers[0];
       float[] workB=reservoir.computeBuffers[1];
       float[] workC=reservoir.computeBuffers[2];
       reservoir.gather(workA);
       reservoir.gather(workB);
       reservoir.gather(workC);
       memory.recallVec(workA, workA);
       reservoir.scatterWritable(workA,location);
       memory.trainVec(workC, workB);
    }

    @Override
    public int weightSize() {
        return 3*reservoir.sizeGather()+reservoir.sizeScatter();
    }

    @Override
    public int buffersRequired() {
        return 3;
    } 
}

