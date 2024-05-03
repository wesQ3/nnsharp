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

    public void Train((NDArray, NDArray)[] trainingInput, int epochs, int batchSize, float trainingRate)
    {
        Console.WriteLine($"begin training.");
        Console.WriteLine($"  {epochs}x, batches of {batchSize} * {trainingRate}");
        var rand = new Random();
        for (int i = 0; i < epochs; i++) {
            rand.Shuffle(trainingInput);
            trainingInput.Chunk(batchSize).ToList()
                .ForEach(b => Update(b, trainingRate));

            Console.WriteLine($"epoch {i,2}/{epochs}: done");
        }
    }

    internal void Update((NDArray x, NDArray y)[] batch, float trainingRate)
    {
        var nablaBias = Biases.Select(b => np.zeros(b.shape)).ToArray();
        var nablaWeight = Weights.Select(w => np.zeros(w.shape)).ToArray();

        foreach (var pair in batch) {
            var (deltaBias, deltaWeight) = Backpropagate(pair.x, pair.y);
            nablaBias = nablaBias.Zip(deltaBias)
                .Select(a => a.First + a.Second).ToArray();
            nablaWeight = nablaWeight.Zip(deltaWeight)
                .Select(a => a.First + a.Second).ToArray();
        }
        Biases = Biases.Zip(nablaBias)
            .Select(a => a.First - (trainingRate / batch.Length) * a.Second)
            .ToArray();
        Weights = Weights.Zip(nablaWeight)
            .Select(a => a.First - (trainingRate / batch.Length) * a.Second)
            .ToArray();
    }

    internal (NDArray[], NDArray[]) Backpropagate(NDArray x, NDArray y)
    {
        throw new NotImplementedException();
    }
}
