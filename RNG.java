package s6regen;

final public class RNG {

    private static final long PHI = 0x9E3779B97F4A7C15L;
    private long s0;
    private long s1;
    private long s2;

    RNG() {
        setSeed(System.nanoTime());
    }

    public long nextLong() {
        return nextLongStafford() ^ nextLongXor();
    }

    public long nextLongXor() {
        final long s0 = this.s0;
        long s1 = this.s1;
        final long result = s0 + s1;
        s1 ^= s0;
        this.s0 = Long.rotateLeft(s0, 55) ^ s1 ^ s1 << 14;
        this.s1 = Long.rotateLeft(s1, 36);
        return result;
    }

    public long nextLongStafford() {
        return staffordMix13(s2 += PHI);
    }
//  Random integer 0 to max inclusive    
    public int nextIntInc(int max){
        long r=max+1L;
        r*=nextLong()&0xffffffffL;
        return (int)(r>>>32); 
    }
    
    public int nextIntInc(int min,int max){
        return nextIntInc(max-min)+min;
    }
    
//  Random integer 0 to bound-1 inclusive
    public int nextIntEx(int bound){
        return (int)(bound*(nextLong()&0xffffffffL)>>>32);
    }
    
    public int nextIntEx(int min,int bound){
        return nextIntEx(bound-min)+min;
    }
    
    public void initPermute(int[] p){
        for(int i=0;i<p.length;i++){
            p[i]=i;
        }
        permute(p);
    }
    
    public void permute(int[] p){
        for(int i=0;i<p.length-1;i++){
            int t=nextIntEx(i,p.length);
            int s=p[t];
            p[t]=p[i];
            p[i]=s;
        }
    }

    public float nextFloat() {
        return (nextLong() & 0x7FFFFFFFFFFFFFFFL) * 1.0842021e-19f;
    }

    public float nextFloatSym() {
        return nextLong() * 1.0842021e-19f;
    }

    public boolean nextBoolean() {
        return nextLong() < 0;
    }

    // Mutation between -1 and 1 
    public float mutate1(long precision) {
        long ra = nextLong();
        int e = 126 - (int) (((ra >>> 32) * precision) >>> 32);
        if (e < 0) {
            return 0f;
        }
        return Float.intBitsToFloat((e << 23) | ((int) ra & 0x807fffff));
    }

    // Mutation between -2 and 2 
    public float mutate2(long precision) {
        long ra = nextLong();
        int e = 127 - (int) (((ra >>> 32) * precision) >>> 32);
        if (e < 0) {
            return 0f;
        }
        return Float.intBitsToFloat((e << 23) | ((int) ra & 0x807fffff));
    }

    // For parameters x>=0, x<=1
    public float mutateX(float x, long precision) {
        float mx = x + mutate1(precision);
        if (mx > 1f) {
            return x;
        }
        if (mx < 0f) {
            return x;
        }
        return mx;
    }

    // For parameters x>=-1, x<=1
    public float mutateXSym(float x, long precision) {
        float mx = x + mutate2(precision);
        if (mx > 1f) {
            return x;
        }
        if (mx < -1f) {
            return x;
        }
        return mx;
    }

    public void setSeed(long seed) {
        s0 = seed;
        s1 = ~seed * 0x9E3779B97F4A7C15L;
        s2 = seed;
        nextLong();
    }

    public static long staffordMix13(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
