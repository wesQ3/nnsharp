using NumSharp;

class Network
{
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

    public static NDArray Sigmoid(NDArray z)
    {
        return 1.0d / (1.0d + np.exp(-z));
    }

    public static NDArray SigmoidPrime(NDArray z)
    {
        var sigmoid = Sigmoid(z);
        return sigmoid * (1.0d - sigmoid);
    }

    public NDArray FeedForward(NDArray input)
    {
        return Biases.Zip(Weights)
            .Aggregate(input, (a, zip) =>
            {
                var weighted = np.dot(zip.Second, a);
                var biased = weighted + zip.First;
                return Sigmoid(biased);
            });
    }

    public void Train((NDArray, NDArray)[] trainingInput, int epochs, int batchSize, double trainingRate)
    {
        Log("begin training.");
        Log($"  {epochs}x, batches of {batchSize} @ {trainingRate}");
        var rand = new Random();
        for (int i = 1; i <= epochs; i++)
        {
            rand.Shuffle(trainingInput);
            trainingInput.Chunk(batchSize).ToList()
                .ForEach(b => Update(b, trainingRate));

            Log($"epoch {i,2}/{epochs}: done");
        }
    }

    internal void Update((NDArray x, NDArray y)[] batch, double trainingRate)
    {
        var nablaBias = Biases.Select(b => np.zeros(b.shape)).ToArray();
        var nablaWeight = Weights.Select(w => np.zeros(w.shape)).ToArray();

        foreach (var pair in batch)
        {
            var (deltaBias, deltaWeight) = Backpropagate(pair.x, pair.y);
            nablaBias = nablaBias.Zip(deltaBias)
                .Select(a => a.First + a.Second).ToArray();
            nablaWeight = nablaWeight.Zip(deltaWeight)
                .Select(a => a.First + a.Second).ToArray();
        }
        Biases = Biases.Zip(nablaBias)
            .Select(a => a.First - trainingRate / batch.Length * a.Second)
            .ToArray();
        Weights = Weights.Zip(nablaWeight)
            .Select(a => a.First - trainingRate / batch.Length * a.Second)
            .ToArray();
    }

    internal (NDArray[], NDArray[]) Backpropagate(NDArray x, NDArray y)
    {
        var nablaBias = Biases.Select(b => np.zeros(b.shape)).ToArray();
        var nablaWeight = Weights.Select(w => np.zeros(w.shape)).ToArray();

        var activation = x;
        List<NDArray> activations = [x];
        List<NDArray> zs = [];

        // feed forward
        foreach (var (bias, weight) in Biases.Zip(Weights))
        {
            var z = np.dot(weight, activation) + bias;
            zs.Add(z);
            activation = Sigmoid(z);
            activations.Add(activation);
        }

        // back prop
        var delta = CostDerivative(activations[^1], y) * SigmoidPrime(zs[^1]);
        nablaBias[^1] = delta;
        nablaWeight[^1] = np.dot(delta, activations[^2].transpose());

        for (int l = 2; l < LayerCount; l++)
        {
            var z = zs[^l];
            var sp = SigmoidPrime(z);
            delta = np.dot(Weights[^(l - 1)].transpose(), delta) * sp;
            nablaBias[^l] = delta;
            nablaWeight[^l] = np.dot(delta, activations[^(l + 1)].transpose());
        }

        return (nablaBias, nablaWeight);
    }

    internal static NDArray CostDerivative(NDArray output_activations, NDArray y)
    {
        return output_activations - y;
    }

    public object ToJson()
    {
        return new
        {
            LayerCount = LayerCount,
            Sizes = Sizes,
            Weights = Weights.Select(w => w.ToArray<double>()).ToArray(),
            Biases = Biases.Select(b => b.ToArray<double>()).ToArray(),
        };
    }

    public Network(int[] sizes, double[][] weights, double[][] biases)
    {
        Sizes = sizes;
        LayerCount = sizes.Length;
        var weightShapes = sizes.SkipLast(1).Zip(sizes.Skip(1));
        Weights = weights.Zip(weightShapes).Select(w =>
        {
            var (x, y) = w.Second;
            return NDArray.FromMultiDimArray<double>(w.First).reshape([y, x]);
        }).ToArray();
        Biases = biases.Select(
            b => NDArray.FromMultiDimArray<double>(b).reshape([-1, 1])
        ).ToArray();
        if (Weights.Length != Biases.Length)
            throw new ArgumentException("array lengths mismatch");
    }
    public static void Log(string s)
    {
        var dt = DateTime.Now.ToString("HH:mm:ss");
        Console.WriteLine($"{dt} {s}");
    }
}
