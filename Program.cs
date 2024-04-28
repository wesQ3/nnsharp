using NumSharp;

var trainLabels = DatasetLoader.LoadIdx("data/train-labels-idx1-ubyte");
var trainImages = DatasetLoader.LoadIdx("data/train-images-idx3-ubyte");
var sample = trainImages[1111];
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
Console.WriteLine("done");
