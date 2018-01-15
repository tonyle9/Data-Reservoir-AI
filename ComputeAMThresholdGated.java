// Associative memory to provide long term memory.
// Reads input address from one gathered source and scatters the result.
// Reads input from another source as an address and stores a further source after
// threshold gating.
package s6regen;

public final class ComputeAMThresholdGated extends Compute{
    
    private final AMThresholdGated memory;
    
    ComputeAMThresholdGated(Reservoir r,int density,float threshold){
        super(r);
        memory=new AMThresholdGated(r.computeSize,density,threshold);
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
    
    @Override
    public void resetHeldState(){
        memory.reset();
    }
    
}
