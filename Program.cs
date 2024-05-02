using NumSharp;

var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
var index = new Random().Next(trainImages.shape[0]);
Console.WriteLine($"pick index: {index}");
sampleArray(trainLabels, index);
sampleArray(trainImages, index);

var nn = new Network([784,15,10]);
testInput(trainImages, index);

trainLabels = DatasetLoader.VectorizeLabels(trainLabels);
var trainingInput = DatasetLoader.MergeDatasets(trainImages, trainLabels);

nn.Train(trainingInput);
Console.WriteLine("done");

void testInput (NDArray target, int index) {
    var sample = target[index];
    // reshape to columns for dot product
    sample = sample.reshape([784]).reshape([-1,1]);
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
