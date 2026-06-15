class Program
{
    static void Main()
    {
        // Point this to your MNIST CSV (label,p0,p1,...,p783)
        Numbers101.RunBinaryDemo(@"C:\Users\user\Documents\archive\mnist_train.csv", labelA: 0, labelB: 1);
    }
}
