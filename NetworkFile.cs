using System.Text.Json;

class NetworkFile {
    public static void Write (Network network) {
        var dt = DateTime.UtcNow.ToString("s");
        File.WriteAllText($"network-{dt}.json",
            JsonSerializer.Serialize(network.ToJson()));
    }

    public static Network Read (string filename) {
        var fs = File.OpenRead(filename);
        return JsonSerializer.Deserialize<Network>(fs)!;
    }

    public static Network ReadLatest () {
        throw new NotImplementedException();
    }

}
