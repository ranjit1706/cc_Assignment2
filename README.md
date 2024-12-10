

---

Link to Docker Image : https://hub.docker.com/r/ranjit1706/wine-quality-testing

## **Steps to Setup and Execute the Project**

### **1. SSH into Instances**
Log into your 4 instances using SSH. Replace `<instance-ip>` with the IP address of each instance.

```bash
ssh -i /path/to/your/private-key.pem ubuntu@<instance-ip>
```

---

### **2. Generate SSH Keys**
On each instance, generate an SSH key pair to enable passwordless communication.

```bash
ssh-keygen -t rsa -N "" -f /home/ubuntu/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
```

Copy the public key from each instance and add it to the `authorized_keys` file of all other instances.

---

### **3. Configure `/etc/hosts`**
On each instance, map the hostnames of all instances in the `/etc/hosts` file.

```bash
sudo vim /etc/hosts
```

Add the following entries (replace `<ip-address>` with actual instance IPs):

```
<ip-address> nn
<ip-address> dd1
<ip-address> dd2
<ip-address> dd3
```

---

### **4. Install Required Software**
Install Java, Maven, and Spark on all instances.

**Install Java:**
```bash
sudo apt update
sudo apt install openjdk-8-jdk -y
```

**Install Maven:**
```bash
sudo apt install maven -y
```

**Install Spark:**
1. Download and extract Spark:
```bash
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
tar -xvzf spark-3.4.1-bin-hadoop3.tgz
```

2. Set environment variables:
```bash
echo "export SPARK_HOME=/home/ubuntu/spark-3.4.1-bin-hadoop3" >> ~/.bashrc
echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

---

### **5. Configure Spark Workers**
Copy the `workers.template` file to `workers` and update it:

```bash
cp $SPARK_HOME/conf/workers.template $SPARK_HOME/conf/workers
vim $SPARK_HOME/conf/workers
```

Add the following lines:
```
localhost
dd1/ip-address
dd2/ip-address
dd3/ip-address
```

---

### **6. Setup Training and Evaluation Directories**
Create `Training` and `Eval` directories on all instances:

```bash
mkdir ~/Training
mkdir ~/Eval
```

Place the Java code files for training and evaluation into these directories.

---

### **7. Run the Training Code**
Use the following command to execute the training code with Spark:

```bash
spark-submit --master spark://<master-ip>:7077 --class com.example.WineQualityEval /home/ubuntu/Training/wine-quality-train-1.0-SNAPSHOT.jar
```

Replace `<master-ip>` with the Spark master instance's IP address.

---

### **8. Create a Docker Image**
Create a Docker image to package your application.

**Dockerfile:**
```dockerfile
# Use the official Spark image as a base image
FROM bitnami/spark:3.4.1

# Set the working directory inside the container
WORKDIR /app

# Copy WineQualityEval (containing the JAR) to the container
COPY WineQualityEval /app/WineQualityEval

# Copy WineQualityPredictionModel to /home/ubuntu
COPY WineQualityPredictionModel /home/ubuntu/WineQualityPredictionModel

# Copy ValidationDataset.csv to /home/ubuntu
COPY ValidationDataset.csv /home/ubuntu/ValidationDataset.csv

# Set the command to run your Spark job
CMD ["spark-submit", "--master", "local", "--class", "com.example.WineQualityEval", "/app/WineQualityEval/target/wine-quality-eval-1.0-SNAPSHOT.jar"]
```

**Build and Push Docker Image:**
```bash
sudo docker build -t ranjit1706/wine-quality-testing:latest .
sudo docker push ranjit1706/wine-quality-testing:latest
```

---

### **9. Pull and Run the Docker Image**
Pull the Docker image on each instance:

```bash
sudo docker pull ranjit1706/wine-quality-testing:latest
```

Run the container:

```bash
sudo docker run ranjit1706/wine-quality-testing:latest
```

---

### **10. Results**
The F1 score of  validation dataset is:

```
F1 Score: 0.8104636591478698
```

---
