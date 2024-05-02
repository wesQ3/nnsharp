using NumSharp;

class Network {
    public int LayerCount;
    public int[] Sizes;
    public NDArray[] Weights;
    public NDArray[] Biases;
    public Network(int[] sizes)
    {
        LayerCount = sizes.Length;
        Sizes = sizes;
        // weights connect between layers, so arrays of sizes(0,1) (1,2) etc
        var zipped = sizes.SkipLast(1).Zip(sizes.Skip(1));
        Weights = zipped
            .Select(zip => np.random.randn([zip.Second, zip.First]))
            .ToArray();
        // 1st layer is for input so skip it
        Biases = sizes.Skip(1).Select(y => np.random.randn(y, 1)).ToArray();
    }

    public static NDArray Sigmoid(NDArray z) {
        return 1/(1 + np.exp(-z));
    }

    public NDArray FeedForward(NDArray input) {
        return Biases.Zip(Weights)
            .Aggregate(input, (a, zip) =>
            {
                var weighted = np.dot(zip.Second, a);
                var biased = weighted + zip.First;
                return Sigmoid(biased);
            });
    }

    public void Train((NDArray, NDArray)[] trainingInput)
    {
        throw new NotImplementedException();
    }
}
