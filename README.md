##  Instruction to use:
---

### 1. SSH into 4 Instances
To SSH into your 4 instances, use the following command for each instance:

```bash
ssh -i /path/to/your/private-key.pem ubuntu@<instance-ip>
```
Repeat this step for all 4 instances.

---

### 2. Generate SSH Keys on All 4 Instances
On each of the instances, execute the following commands:

```bash
ssh-keygen -t rsa -N "" -f /home/ubuntu/.ssh/id_rsa
cd ~/.ssh
ls -lrt  # To see the generated keys
cat ~/.ssh/id_rsa.pub  # To display the public key
```

Make sure to note down the output of the `cat ~/.ssh/id_rsa.pub` command, as you will need to copy this public key to the other instances.

---

### 3. Add Public Keys to `authorized_keys` on All 4 Instances
On each instance, execute the following:

1. Open the `authorized_keys` file in `vim`:

```bash
vim ~/.ssh/authorized_keys
```

2. Copy and paste the public keys from all 4 instances into this file. Each key should be on a new line.

---

### 4. Edit `/etc/hosts` on All 4 Instances
On each instance, add the following entries to `/etc/hosts` to define hostnames for each instance:

1. First, navigate to the home directory:

```bash
cd
```

2. Open the `/etc/hosts` file in `vim`:

```bash
sudo vim /etc/hosts
```

3. Add the following lines:

```
<ip-address> nn
<ip-address> dd1
<ip-address> dd2
<ip-address> dd3
```

Replace `<ip-address>` with the actual IP addresses of your instances.

---

### 5. Install Java, Maven, and Spark 3.4.1 on All 4 Instances
On each instance, install Java, Maven, and Spark as follows:

Install Java:
```bash
sudo apt update
sudo apt install openjdk-11-jdk -y
java -version  # To verify installation
```

Install Maven:
```bash
sudo apt install maven -y
mvn -version  # To verify installation
```

Install Spark 3.4.1:

1. Download Spark from the official link (e.g., Spark 3.4.1):
```bash
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.2.tgz
```

2. Extract Spark:
```bash
tar -xvzf spark-3.4.1-bin-hadoop3.2.tgz
```

3. Set Spark environment variables:
```bash
echo "export SPARK_HOME=/home/ubuntu/spark-3.4.1-bin-hadoop3.2" >> ~/.bashrc
echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

---

### 6. Copy `workers.template` to `workers`
Now, copy the `workers.template` file to `workers`:

```bash
cp $SPARK_HOME/conf/workers.template $SPARK_HOME/conf/workers
```

---

### 7. Edit `workers` File
Open the `workers` file in `vim` and add the following lines after `localhost`:

```bash
vim $SPARK_HOME/conf/workers
```

Add the following (replace `dd1/ip-address`, `dd2/ip-address`, `dd3/ip-address` with the actual IP addresses of your instances):

```
localhost
dd1/ip-address
dd2/ip-address
dd3/ip-address
```

---

### 8. Install Spark 3.4.1, Java, and Maven
Ensure that Spark, Java, and Maven are installed on all the instances as explained above.

---

### 9. Create Folders for Training and Eval and Place Java Codes
On each instance, create two folders named `Training` and `Eval`:

```bash
mkdir ~/Training
mkdir ~/Eval
```

Place the respective Java code files (for training and evaluation) inside these folders.

---

### 10. Run the Training Code in Parallel Using `spark-submit`
To run your training code using `spark-submit` on all instances in parallel, use the following command:

```bash
spark-submit --master spark://<master-ip>:7077 --class com.example.WineQualityEval /home/ubuntu/Training/wine-quality-train-1.0-SNAPSHOT.jar
```

Replace `<master-ip>` with the actual IP address of the Spark master instance.

---

### 11. Create a Docker Image
Once your code is ready and everything is configured, create a Docker image. Here’s the `Dockerfile` we previously worked on:

```dockerfile
# Use the official Spark image as a base image
FROM bitnami/spark:3.4.1

# Set the working directory inside the container
WORKDIR /app

# Copy WineQualityEval (containing the JAR) to the container
COPY WineQualityEval /app/WineQualityEval

# Copy WineQualityPredictionModel to /home/ubuntu (matching the JAR's expected path)
COPY WineQualityPredictionModel /home/ubuntu/WineQualityPredictionModel

# Copy ValidationDataset.csv to /home/ubuntu (matching the JAR's expected path)
COPY ValidationDataset.csv /home/ubuntu/ValidationDataset.csv

# Set the command to run your Spark job using spark-submit
CMD ["spark-submit", "--master", "local", "--class", "com.example.WineQualityEval", "/app/WineQualityEval/target/wine-quality-eval-1.0-SNAPSHOT.jar"]
```

To build the Docker image, run:

```bash
sudo docker build -t ranjitkotamraju/wine-quality-eval:latest .
```

---

### 12. Push the Docker Image
Once the image is built, you can push it to Docker Hub:

```bash
sudo docker push ranjitkotamraju/wine-quality-eval:latest
```

Make sure you're logged in to Docker Hub using:

```bash
sudo docker login
```

---

### 13. Pull the Docker Image on Desired Instances
To pull the image on the desired instances, run the following command on each instance:

```bash
sudo docker pull ranjitkotamraju/wine-quality-eval:latest
```

---

### 14. Run the Docker Container
Once the image is pulled, you can run it with:

```bash
sudo docker run -it --entrypoint /bin/bash ranjitkotamraju/wine-quality-eval:latest
```

This will start the container and open a bash shell inside the container. You can then execute any further commands required.

---

### 15. Run the Spark Job in Docker
To run the Spark job within the Docker container, use the following command:

```bash
spark-submit --master spark://<master-ip>:7077 --class com.example.WineQualityEval /app/WineQualityEval/target/wine-quality-eval-1.0-SNAPSHOT.jar
