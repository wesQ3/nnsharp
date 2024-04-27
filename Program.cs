using NumSharp;
using System;
using System.IO;

Console.WriteLine("Hello, World!");

var fs = File.OpenRead("data/train-labels-idx1-ubyte");
for (int i = 0; i < 16; i++) {
    var b = fs.ReadByte();
    Console.WriteLine(b.ToString());
}

Console.WriteLine("done reading");
