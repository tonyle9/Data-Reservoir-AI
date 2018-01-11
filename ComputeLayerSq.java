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
        int mask = reservoir.computeSize - 1;
        int len = reservoir.computeSize * density;
        float[] workA = reservoir.computeBuffers[0];
        float[] workB = reservoir.computeBuffers[1];
        reservoir.gather(workA);
        Arrays.fill(workB, 0f);
        for (int i = 0; i < len; i++) {
            int idx = i & mask;
            if (idx == 0) {
                WHT.fastRP(workA, reservoir.hashIndex++);
            }
            workB[idx] += workA[idx] * workA[idx] * reservoir.weights[reservoir.weightIndex++];
        }
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
