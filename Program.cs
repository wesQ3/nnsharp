using NumSharp;
using System;
using System.Buffers.Binary;
using System.IO;
using System.Reflection;

// var filename = "data/train-labels-idx1-ubyte";
var filename = "data/train-images-idx3-ubyte";
Console.WriteLine($"read from {filename}");

var fs = File.OpenRead(filename);
Console.WriteLine($"file size {fs.Length}");
var br = new BinaryReader(fs);
var magic = br.ReadBytes(4);
Console.WriteLine($"magic: {Convert.ToHexString(magic)}");

var dimensions = (int) magic[3];
Console.WriteLine($"dimensions: {dimensions}");

int[] sizes = new int[dimensions];
for (int i = 0; i < dimensions; i++) {
    var bigend = br.ReadBytes(4);
    sizes[i] = (int) BinaryPrimitives.ReadUInt32BigEndian(bigend);
    Console.WriteLine($"size {i}: {sizes[i]}");
}

var total = sizes.Aggregate((acc, i) => acc * i);
var body = br.ReadBytes(total);
var array = np.array(body);
var shape = new Shape(sizes);
array = array.reshape(shape);
Console.WriteLine("new shape: "+string.Join(",", array.shape));
var sample = array[1111];
Console.WriteLine("sample shape: "+string.Join(",", sample.shape));
Console.WriteLine($"sample length: {sample.size}");
for (var i = 0; i < sample.shape[0]; i++) {
    var line = sample[i].ToByteArray();
    var str = string.Join(",", line.Select(b =>  $"{b,3}"));
    Console.WriteLine($"{i,2}: {str}");
}
Console.WriteLine("done");
