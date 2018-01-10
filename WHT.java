package s6regen;
// Minimum vector lenght=16, vector length must be a power of 2 (16,32,64,128..)
// Includes functions Java can't autovectorize so well.  Though I could have
// left out signOfNative.  WHT=Walsh Hadamard transform.
// OS=Linux AMD64, Library to install libwth.so. Falls back to pure Java otherwise. 
public class WHT {

    static final boolean useNative;

    static native void whtNative(float[] vec);

    static native void signFlipNative(float[] vec, long h);

    static native void signOfNative(float[] sign, float[] x);

    static void whtJava(float[] vec) {
        int i, j, hs = 1, n = vec.length;
        while (hs < n) {
            i = 0;
            while (i < n) {
                j = i + hs;
                while (i < j) {
                    float a = vec[i];
                    float b = vec[i + hs];
                    vec[i] = a + b;
                    vec[i + hs] = a - b;
                    i += 1;
                }
                i += hs;
            }
            hs += hs;
        }
        float sc = 1f / (float) Math.sqrt(n);
        for (i = 0; i < n; i++) {
            vec[i] *= sc;
        }
    }

    public static void wht(float[] vec) {
        if (useNative) {
            whtNative(vec);
        } else {
            whtJava(vec);
        }
    }

    static void signFlipJava(float[] vec, long h) {
        h = h * 2862933555777941757L + 3037000493L;
        for (int i = 0; i < vec.length; i++) {
            h = h * 2862933555777941757L + 3037000493L;
            int x = (int) (h >>> 32) & 0x80000000;  // select sign flag bit
            vec[i] = Float.intBitsToFloat(x ^ Float.floatToRawIntBits(vec[i]));  // xor top bit
        }
    }

//  Randomly flips the sign of elements of vec using h as a pseudorandom seed.    
    public static void signFlip(float[] vec, long h) {
        if (useNative) {
            signFlipNative(vec, h);
        } else {
            signFlipJava(vec, h);
        }
    }

//  Gets sign of elements of x.
    static void signOfJava(float[] sign, float[] x) {
        int one = Float.floatToRawIntBits(1f);
        for (int i = 0; i < sign.length; i++) {
            sign[i] = Float.intBitsToFloat(one | (Float.floatToRawIntBits(x[i]) & 0x80000000));
        }
    }

    public static void signOf(float[] sign, float[] x) {
        if (useNative) {
            signOfNative(sign, x);
        } else {
            signOfJava(sign, x);
        }
    }

//  Fast random projection.  
    public static void fastRP(float[] vec, long h) {
        signFlip(vec, h);
        wht(vec);
    }

    static {
        boolean flag = true;
        try {
            System.loadLibrary("wht");
        } catch (UnsatisfiedLinkError e) {
            flag = false;
        }
        System.out.println("Using native WHT: " + flag);
        useNative = flag;
    }

}
