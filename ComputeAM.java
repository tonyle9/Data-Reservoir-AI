// Associative memory to provide long term memory.
// Reads input address from one gathered source and scatters the result.
// Reads input from another source as an address and stores a further source.
package s6regen;

public final class ComputeAM extends Compute{
    
    private final AM memory;
    
    ComputeAM(Reservoir r,int density){
        super(r);
        memory=new AM(r.computeSize,density);
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
       reservoir.scatter(workA);
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
