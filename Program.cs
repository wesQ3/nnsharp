using NumSharp;
const int imageSize = 28 * 28;

if (args.Contains("--network"))
{
    var index = Array.IndexOf(args, "--network");
    if (index == -1 || index - 1 == args.Length)
        Bail();

    var filename = args[index + 1];
    var nn = NetworkFile.Read(filename);
    Read(nn);
}
else if (args.Contains("--train"))
    Train();
else if (args.Contains("--latest"))
    ReadLatest();
else if (args.Contains("--roundtrip"))
    Roundtrip();
else
    ReadLatest();

void Bail()
{
    Console.WriteLine("usage: --train | --latest | --network <file>");
    Environment.Exit(1);
}

void Roundtrip()
{
    var outNet = new Network([5, 3, 2]);
    NetworkFile.Write(outNet, "roundtrip.json");
    var inNet = NetworkFile.Read("roundtrip.json");
    Console.WriteLine($"{outNet == inNet}");
    var testInput = np.random.randn([5, 1]);
    var preOutput = outNet.FeedForward(testInput);
    var postOutput = inNet.FeedForward(testInput);
    Console.WriteLine($"{preOutput == postOutput}");
}

void Read(Network nn)
{
    var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
    var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
    var index = new Random().Next(trainImages.shape[0]);
    Console.WriteLine($"pick index: {index}");
    sampleArray(trainLabels, index);
    sampleArray(trainImages, index);
    testInput(trainImages, index, nn);
    Console.WriteLine("done");
}

void ReadLatest()
{
    var nn = NetworkFile.ReadLatest();
    Read(nn);
}

void Train()
{
    var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
    var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
    var index = new Random().Next(trainImages.shape[0]);
    Console.WriteLine($"pick index: {index}");
    sampleArray(trainLabels, index);

    var nn = new Network([imageSize, 15, 10]);
    testInput(trainImages, index, nn);
    trainImages = trainImages.reshape([trainImages.shape[0], imageSize, 1]);

    trainLabels = DatasetLoader.VectorizeLabels(trainLabels);
    var trainingInput = DatasetLoader.MergeDatasets(trainImages, trainLabels);

    nn.Train(trainingInput, 30, 10, 3.0d);
    NetworkFile.Write(nn);
    testInput(trainImages, index, nn);
    Console.WriteLine("done");
}

void testInput(NDArray target, int index, Network nn)
{
    var sample = target[index];
    // reshape to columns for dot product
    // shape (imageSize, 1)
    sample = sample.reshape([imageSize, 1]);
    var result = nn.FeedForward(sample);
    Console.WriteLine(" 0: " + string.Join(",", result.ToArray<Double>()));
}

void sampleArray(NDArray target, int index)
{
    var sample = target[index];
    Console.WriteLine($"sample length: {sample.size}");
    if (np.isscalar(sample))
    {
        Console.WriteLine(" 0: " + string.Join(",", sample.ToByteArray()));
    }
    else
    {
        string map = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
        Console.WriteLine("sample shape: " + string.Join(",", sample.shape));
        for (var i = 0; i < sample.shape[0]; i++)
        {
            var line = sample[i].ToByteArray();
            var str = string.Join(" ", line.Select(b => map[b * (map.Length - 1) / 255]));
            Console.WriteLine($"{i,2}: {str}");
        }
    }
}
