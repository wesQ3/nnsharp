using System.Text.Json;

class NetworkFile
{
    public static void Write(Network network, string filename)
    {
        File.WriteAllText(filename,
            JsonSerializer.Serialize(network.ToJson()));
    }

    public static void Write(Network network)
    {
        var dt = DateTime.UtcNow.ToString("s");
        Write(network, $"network-{dt}.json");
    }

    public static Network Read(string filename)
    {
        Console.WriteLine($"read network {filename}");
        var fs = File.OpenRead(filename);
        var nd = JsonSerializer.Deserialize<NetworkData>(fs)!;
        return new Network(nd.Sizes, nd.Weights, nd.Biases);
    }

    public static Network ReadLatest()
    {
        var dir = Directory.GetCurrentDirectory();
        var file = Directory.EnumerateFiles(dir, "network-*.json")
            .OrderByDescending(name => name)
            .FirstOrDefault()
            ?? throw new FileNotFoundException("no network files yet. train() first");

        return Read(file);
    }
}

class NetworkData
{
    public double[][] Weights { get; set; }
    public double[][] Biases { get; set; }
    public int[] Sizes { get; set; }
}
