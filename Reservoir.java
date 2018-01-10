package s6regen;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;

public class Reservoir implements Serializable {

    private final int computeSize;
    private final int reservoirSize;
    private final int inputSize;
    private final int writableSize;
    private final int outputSize;

    int hashIndex;
    int weightIndex;

    private int weightSize;
    float[] weights;
    private transient float[] reservoir;
    private transient float[][] computeBuffers;
    private transient RNG rng;
    private final ArrayList<Compute> list;

    Reservoir(int computeSize, int reservoirSize, int inputSize, int writableSize, int outputSize) {
        assert computeSize >= 16 : "Requirement for WTH class";
        assert (computeSize & (computeSize - 1)) == 0 : "Power of 2 for WHT class";
        assert reservoirSize % computeSize == 0 : "Make multiple of computeSize";
        assert inputSize % computeSize == 0 : "Make multiple of computeSize";
        assert writableSize % computeSize == 0 : "Make multiple of computeSize";

        this.computeSize = computeSize;
        this.reservoirSize = reservoirSize;
        this.inputSize = inputSize;
        this.writableSize = writableSize;
        this.outputSize = outputSize;
        list = new ArrayList<>();
    }

    Reservoir(Reservoir r) {
        computeSize = r.computeSize;
        reservoirSize = r.reservoirSize;
        inputSize = r.inputSize;
        writableSize = r.weightSize;
        outputSize = r.outputSize;
        list = new ArrayList<>(r.list);
    }

    public void addComputeUnit(Compute c) {
        list.add(c);
    }

    public void prepareForUse() {
        int bN = 0;
        for (Compute c : list) {
            weightSize += c.weightSize();
            if (bN < c.buffersRequired()) {
                bN = c.buffersRequired();
            }
        }
        if (weights != null) {
            weights = new float[weightSize];
        }
        reservoir = new float[reservoirSize];
        computeBuffers = new float[bN][computeSize];
        rng = new RNG();
    }

    public void computeAll() {
        hashIndex = 0;
        weightIndex = 0;
        for (Compute c : list) {
            c.compute();
        }
    }

    public void setInput(float[] input) {
        System.arraycopy(input, 0, reservoir, 0, inputSize);
    }

    public void getOutput(float[] output) {
        System.arraycopy(reservoir, inputSize + writableSize, output, 0, outputSize);
    }

    public void copyWeightsMutateFrom(Reservoir r, long mutatePrecision) {
        for (int i = 0; i < weightSize; i++) {
            weights[i] = rng.mutateXSym(r.weights[i], mutatePrecision);
        }
    }

    void gather(float[] g, int weightIndex) {
        int mask = computeSize - 1;
        for (int i = 0; i < computeSize; i++) {
            g[i] = reservoir[i] * weights[weightIndex++];
        }
        for (int i = computeSize; i < reservoirSize; i++) {
            if ((i & mask) == 0) {
                WHT.fastRP(g, hashIndex++);
            }
            g[i & mask] = +reservoir[i] * weights[weightIndex++];
        }
        WHT.fastRP(g, hashIndex++);
    }

    void scatter(float[] s, int weightIndex) {
        int mask = computeSize - 1;
        for (int i = inputSize + writableSize; i < reservoirSize; i++) {
            if ((i & mask) == 0) {
                WHT.fastRP(s, hashIndex++);
            }
            reservoir[i] += s[i & mask] * weights[weightIndex++];
        }
    }

    void scatterWritable(float[] s, int location) {
        System.arraycopy(s, 0, reservoir, inputSize + location, s.length);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        prepareForUse(); //Set up all the buffers and working arrays
    }
}
