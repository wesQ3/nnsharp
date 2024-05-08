using NumSharp;
const int imageSize = 28*28;

roundtrip();

void roundtrip () {
    var outNet = new Network([5,3,2]);
    NetworkFile.Write(outNet, "roundtrip.json");
    var inNet = NetworkFile.Read("roundtrip.json");
    Console.WriteLine($"{outNet == inNet}");
    var testInput = np.random.randn([5, 1]);
    var preOutput = outNet.FeedForward(testInput);
    var postOutput = inNet.FeedForward(testInput);
    Console.WriteLine($"{preOutput == postOutput}");

}

void latest() {
    var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
    var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
    var index = new Random().Next(trainImages.shape[0]);
    Console.WriteLine($"pick index: {index}");
    sampleArray(trainLabels, index);
    var nn = NetworkFile.ReadLatest();
    testInput(trainImages, index, nn);
    Console.WriteLine("done");
}

void train() {
    var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
    var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
    var index = new Random().Next(trainImages.shape[0]);
    Console.WriteLine($"pick index: {index}");
    sampleArray(trainLabels, index);

    var nn = new Network([imageSize,15,10]);
    testInput(trainImages, index, nn);
    trainImages = trainImages.reshape([trainImages.shape[0], imageSize]);

    trainLabels = DatasetLoader.VectorizeLabels(trainLabels);
    var trainingInput = DatasetLoader.MergeDatasets(trainImages, trainLabels);

    nn.Train(trainingInput, 30, 10, 3.0f);
    NetworkFile.Write(nn);
    testInput(trainImages, index, nn);
    Console.WriteLine("done");
}

void testInput (NDArray target, int index, Network nn) {
    var sample = target[index];
    // reshape to columns for dot product
    // shape (imageSize, 1)
    sample = sample.reshape([imageSize]).reshape([-1,1]);
    var result = nn.FeedForward(sample);
    Console.WriteLine(" 0: " + string.Join(",", result.ToArray<Double>()));
}

void sampleArray (NDArray target, int index) {
    var sample = target[index];
    Console.WriteLine($"sample length: {sample.size}");
    if (np.isscalar(sample)) {
        Console.WriteLine(" 0: " + string.Join(",", sample.ToByteArray()));
    } else {
        Console.WriteLine("sample shape: " + string.Join(",", sample.shape));
        for (var i = 0; i < sample.shape[0]; i++)
        {
            var line = sample[i].ToByteArray();
            var str = string.Join(",", line.Select(b => $"{b,3}"));
            Console.WriteLine($"{i,2}: {str}");
        }
    }
}
