using NumSharp;
using System;
using System.Buffers.Binary;
using System.IO;
using System.Reflection;

var filename = "data/train-images-idx3-ubyte";
Console.WriteLine($"read from {filename}");

var fs = File.OpenRead(filename);
Console.WriteLine($"file size {fs.Length}");
var br = new BinaryReader(fs);
var magic = br.ReadBytes(4);
Console.WriteLine($"magic: {Convert.ToHexString(magic)}");

var dimensions = (int) magic[3];
Console.WriteLine($"dimensions: {dimensions}");

uint[] sizes = new uint[dimensions];
for (int i = 0; i < dimensions; i++)
{
    var bigend = br.ReadBytes(4);
    sizes[i] = BinaryPrimitives.ReadUInt32BigEndian(bigend);
    Console.WriteLine($"size {i}: {sizes[i]}");
}
Console.WriteLine("done");
