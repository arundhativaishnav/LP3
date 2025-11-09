package DAA;


public class fibonacciSeries {
    public static void main(String[] args) {
        int n = 10; // Number of terms
        System.out.println("Fibonacci series up to " + n + " terms:");

        for (int i = 0; i < n; i++) {
            System.out.print(fibIterative(i) + " ");
        }
    }

    // Iterative Fibonacci
    static int fibIterative(int n) {
        if (n <= 1)
            return n;

        int a = 0, b = 1, c = 0;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
  // Recursive Fibonacci
    static int fibRecursive(int n) {
        if (n <= 1)
            return n;
        return fibRecursive(n - 1) + fibRecursive(n - 2);
    }
}