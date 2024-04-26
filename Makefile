# List of MNIST dataset URLs
URLS := \
  https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz \
  https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz \
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz \
  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Target directory for downloaded and uncompressed files
DATA_DIR := data

# Default target
all: $(DATA_DIR)
	$(MAKE) download_and_unzip

# Create data directory
$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

# Download and unzip files
download_and_unzip:
	@$(foreach url,$(URLS), \
		wget -c $(url) -O $(DATA_DIR)/$(notdir $(url)) && \
		gzip -d $(DATA_DIR)/$(notdir $(url));)

# Clean up the data directory
clean:
	rm -rf $(DATA_DIR)