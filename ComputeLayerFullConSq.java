package s6regen;
// Fully connected layer
// Calculates a single neural network layer with x square activation function.

public class ComputeLayerFullConSq extends Compute {

    ComputeLayerFullConSq(Reservoir r) {
        super(r);
    }

    @Override
    public void compute() {
        int cs = reservoir.computeSize;
        float[] workA = reservoir.computeBuffers[0];
        float[] workB = reservoir.computeBuffers[1];
        float[] wt = reservoir.weights;    // get a local copy as an optimization
        reservoir.gather(workA);
        int wtIdx = reservoir.weightIndex; // must get after gather
        WHT.fastRP(workA, reservoir.hashIndex++);
        for (int i = 0; i < cs; i++) {
            workB[i] = workA[0] * wt[wtIdx++];
        }
        for (int i = 1; i < cs; i++) {
            for (int j = 0; j < cs; j++) {
                workB[j] += workA[i] * wt[wtIdx++];
            }
        }
        reservoir.weightIndex = wtIdx;    // must set before scatter
        VecOps.multiply(workB, workB, workB);
        reservoir.scatter(workB);
    }

    @Override
    public int weightSize() {
        return reservoir.sizeGather() + reservoir.sizeScatter() + reservoir.computeSize * reservoir.computeSize;
    }

    @Override
    public int buffersRequired() {
        return 2;
    }

}
