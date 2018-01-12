package s6regen;

import java.util.Arrays;
// Calculates a single neural network layer with x square activation function.
// Sparsity inducing and no problem with evolution based training.

public class ComputeLayerSq extends Compute {

    private final int density;

    ComputeLayerSq(Reservoir r, int density) {
        super(r);
        assert density > 0 : "Density at least 1";
        this.density = density;
    }

    @Override
    public void compute() {
        int cs = reservoir.computeSize;
        int len = reservoir.computeSize * density;
        float[] workA = reservoir.computeBuffers[0];
        float[] workB = reservoir.computeBuffers[1];
        float[] wt = reservoir.weights;    // get a local copy as an optimization
        reservoir.gather(workA);
        int wtIdx = reservoir.weightIndex; // must get after gather
        WHT.fastRP(workA, reservoir.hashIndex++);
        for (int i = 0; i < cs; i++) {
            workB[i] = workA[i] * wt[wtIdx++];
        }
        for (int i = cs; i < len; i += cs) {
            WHT.fastRP(workA, reservoir.hashIndex++);
            for (int j = 0; j < cs; j++) {
                workB[j] += workA[j] * wt[wtIdx++];
            }
        }
        reservoir.weightIndex = wtIdx;    // must set before scatter
        VecOps.multiply(workB, workB, workB);
        reservoir.scatter(workB);
    }

    @Override
    public int weightSize() {
        return reservoir.sizeGather() + reservoir.sizeScatter() + density * reservoir.computeSize;
    }

    @Override
    public int buffersRequired() {
        return 2;
    }

}
