package s6regen;

class VecOps {

    final static float MIN_SQ = 1e-20f;

    static void multiply(float[] rVec, float[] x, float[] y) {
        for (int i = 0; i < rVec.length; i++) {
            rVec[i] = x[i] * y[i];
        }
    }

    static void multiplyAddTo(float[] rVec, float[] x, float[] y) {
        for (int i = 0; i < rVec.length; i++) {
            rVec[i] += x[i] * y[i];
        }
    }

    // x-y
    static void subtract(float[] rVec, float[] x, float[] y) {
        for (int i = 0; i < rVec.length; i++) {
            rVec[i] = x[i] - y[i];
        }
    }

    static void add(float[] rVec, float[] x, float[] y) {
        for (int i = 0; i < rVec.length; i++) {
            rVec[i] = x[i] + y[i];
        }
    }

    static void scale(float[] rVec, float[] x, float s) {
        for (int i = 0; i < rVec.length; i++) {
            rVec[i] = x[i] * s;
        }
    }

    // reduce the magnitude by t, if the magnitude is reduced below 0 it is made 0.
    // with t=1, 1.5 becomes 0.5, -2.5 becomes -1.5, .9 becomes 0 etc.
    static void truncate(float[] rVec, float[] x, float t) {
        for (int i = 0; i < rVec.length; i++) {
            int f = Float.floatToRawIntBits(x[i]);
            int s = f & 0x80000000;  // get sign bit
            float m = Float.intBitsToFloat(f & 0x7fffffff) - t; //abs(x[i])-t
            if (m < 0f) {
                m = 0f;
            }
            rVec[i] = Float.intBitsToFloat(Float.floatToRawIntBits(m) | s); // put sign back in
        }
    }

    static float sumSq(float[] vec) {
        float sum = 0f;
        for (int i = 0; i < vec.length; i++) {
            sum += vec[i] * vec[i];
        }
        return sum;
    }

    // Assuming each elememt of is from a Gaussian distribution of zero mean
    // adjust the variance of each element to 1.
    static void adjust(float[] rVec, float[] x) {
        float adj = 1f / (float) Math.sqrt((sumSq(x) / x.length) + MIN_SQ);
        scale(rVec, x, adj);
    }
}
