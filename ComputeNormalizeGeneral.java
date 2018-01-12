// Rescale the general section of the reservoir to a standard vector length
package s6regen;

public final class ComputeNormalizeGeneral extends Compute {

    public ComputeNormalizeGeneral(Reservoir r) {
        super(r);
    }

    @Override
    public void compute() {
        reservoir.normalize();
    }

    @Override
    public int weightSize() {
        return 0;
    }

    @Override
    public int buffersRequired() {
        return 0;
    }
}
