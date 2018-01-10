package s6regen;

import java.util.Arrays;

public class ComputeLayer extends Compute {
    
    private final int density;
    
    ComputeLayer(Reservoir r, int density){
        super(r);
        assert density>0:"Density at least 1";
        this.density=density;
    }

    @Override
    public void compute() {
       int mask=reservoir.computeSize-1;
       int len=reservoir.computeSize*density;
       float[] workA=reservoir.computeBuffers[0];
       float[] workB=reservoir.computeBuffers[1];
       reservoir.gather(workA);
       Arrays.fill(workB, 0f);
       for(int i=0;i<len;i++){
           if((i & mask)==0) WHT.fastRP(workA, reservoir.hashIndex++);
           workB[i & mask]+=workA[i & mask]*reservoir.weights[reservoir.weightIndex++];
       }
       reservoir.scatter(workB);
    }

    @Override
    public int weightSize() {
        return reservoir.sizeGather()+reservoir.sizeScatter()+density*reservoir.computeSize;
    }

    @Override
    public int buffersRequired() {
        return 2;
    }
    
}
