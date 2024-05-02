using NumSharp;
using System.Buffers.Binary;

class DatasetLoader
{
    public static NDArray LoadIdx(string filename)
    {
        Console.WriteLine($"read from {filename}");
        var fs = File.OpenRead(filename);
        Console.WriteLine($"file size {fs.Length}");
        var br = new BinaryReader(fs);
        var magic = br.ReadBytes(4);
        Console.WriteLine($"magic: {Convert.ToHexString(magic)}");

        var dimensions = (int)magic[3];
        Console.WriteLine($"dimensions: {dimensions}");

        int[] sizes = new int[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            var bigend = br.ReadBytes(4);
            sizes[i] = (int)BinaryPrimitives.ReadUInt32BigEndian(bigend);
            Console.WriteLine($"size {i}: {sizes[i]}");
        }

        var total = sizes.Aggregate((acc, i) => acc * i);
        var body = br.ReadBytes(total);
        var array = np.array(body);
        var shape = new Shape(sizes);
        array = array.reshape(shape);
        Console.WriteLine("new shape: " + string.Join(",", array.shape));
        return array;
    }

    public static (NDArray, NDArray)[] MergeDatasets(NDArray a, NDArray b)
    {
        if (a.shape[0] != b.shape[0])
            throw new ArgumentException("whats all this");

        return Enumerable.Range(0, a.shape[0])
            .Select(i => (a[i], b[i]))
            .ToArray();
    }

    internal static NDArray VectorizeLabels(NDArray trainLabels)
    {
        List<NDArray> vectors = [];
        foreach (var label in trainLabels) {
            var z = np.zeros([10,1]);
            byte unbox = (byte)label;
            z[unbox] = 1.0;
            vectors.Add(z);
        }
        return np.stack(vectors.ToArray()).reshape(new Shape(trainLabels.size, 10));
        // NotSupported
        // return trainLabels.GetData<int>()
        //     .Select(val => { var z = np.zeros([10,1]); z[val] = 1.0; return z; })
        //     .ToArray();
    }
}
