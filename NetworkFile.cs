using System.Text.Json;

class NetworkFile {
    public static void Write (Network network) {
        var dt = DateTime.UtcNow.ToString("s");
        File.WriteAllText($"network-{dt}.json",
            JsonSerializer.Serialize(network.ToJson()));
    }

    public static Network Read (string filename) {
        Console.WriteLine($"read network {filename}");
        var fs = File.OpenRead(filename);
        return JsonSerializer.Deserialize<Network>(fs)!;
    }

    public static Network ReadLatest () {
        var dir = Directory.GetCurrentDirectory();
        var file = Directory.EnumerateFiles(dir, "network-*.json")
            .OrderByDescending(name => name)
            .FirstOrDefault()
            ?? throw new FileNotFoundException("no network files yet. train() first");

        return Read(file);
    }
}
