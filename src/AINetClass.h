#pragma once

class AINetClass
{
public:
	//variables

	//functions
	AINetClass();
	~AINetClass();
	//AINetClass(const AINetClass &oldClass); 
	unsigned int NUMNODES();
	unsigned int NUMINPUTNODES();
	unsigned int NUMREALINPUTNODES();
	unsigned int NUMOUTPUTNODES();
	unsigned int NUMHIDDENNODES();
	unsigned int SizeOfArray();
	unsigned int getMaxIterations();
	unsigned int Counter(bool bIncrease = false);
	unsigned int CurrentTrainingDataRow();
	unsigned int getActivationFunction();
	unsigned int getActivationFunction(unsigned int tmpNodeID);
	unsigned int getNumberOfNodesInLayer(int tmpLayer);
	unsigned int getNumberOfLayers(bool bOnlyHidden = false);
	unsigned int getLayerStart(int tmpLayer, bool falseForLayerEnd = true);
	unsigned int LoadTrainingDataFile();
	double LearningRate();
	size_t TrainingDataColumns();
	std::string TrainingDataColumnName(unsigned int tmpColumn, bool shortList=false);
	unsigned int getTrainingDataRowsMax();	
	unsigned int getMaximumNodesLayer(bool bGetMaximumNodes = false);
	unsigned int getLayerByNode(unsigned int itmpNode);
	bool continueCalculation();
	bool IsTrainingRestart();
	bool IsLastLayer(unsigned int tmpLayer);
	bool getOptionStatus();
	bool setNumInputNodes(unsigned int tmpInputNodes);
	bool setNumOutputNodes(unsigned int tmpOutputNodes);
	bool setTimePrevRows(int tmpPrevRows);
	bool setTimeInputColumns(unsigned int tmpPrevCols);
	bool setMaxIterations(unsigned int tmpMaxIterations);
	bool setLearningRate(double tmpLearningRate);
	bool setTrainingDataRowsMax(unsigned int tmpMaxRows);
	bool setNumberOfHiddenLayers(unsigned int tmpHiddenLayers, unsigned int tmpNodesinHiddenLayer = 1);
	bool setNumberOfNodesinLayer(int tmpLayer, unsigned int tmpNumberOfNodes);
	bool resetCounter();
	bool IsNetworkReady();
	bool autoGenerateInternalNetwork();
	double NodeFunction(double weightedInput, unsigned int currentNodeID, bool derivative = false);
	double updateWeights();
	double getNodeValue(unsigned int tmpNode);
	void TrainingDataColumnPush_Back(std::string tmpString);
	void activateNetwork();
	double calculateErrorMSE(int iLayer);
	void connectNodes(bool bFullyConnected = true, unsigned int iRandSeed = 0);
	void closeTrainingDataFile(std::ifstream &ptrDataFile);
	void combineNetworks(AINetClass& ptrAiNetClass, std::mutex & ptrMutex, unsigned int iNumber=0);
	void createNetwork(std::vector<std::string> tmpsNetwork);
	void createNetwork(std::vector<unsigned int> tmpviNetwork);
	void displayIO(double sumOfSquaredErrors);
	void displayWeights();
	void displayStatus();
	void displayAllNodes(double sumOfSquaredErrors);
	void initialize();
	bool openTrainingDataFile(std::ifstream &ptrDataFile);
	void saveResultingNetwork(unsigned int iNumber = 0);
	void setActivationFunction(unsigned int typeOfActivationFunction, unsigned int specificLayer = 0);
	void setDataFileName(std::string strFileName);
	void setInternalName(std::string strIntName);
	void setOptionAutoGenerate(bool bAutoGenerate);
	void setOptionCSV(bool bSetGerman);
	void setOptionDisplayAllNodes(bool bDisplayAll);
	void setOptionIO(bool bSetIO);
	void setOptionNoDeep(bool bSetNoDeep);
	void setOptionShuffle(bool bSetShuffle);
	void setOptionSilent(bool bSilent);
	void setOptionStatus(bool bSetStatus);
	void setOptionWeight(bool bSetWeight);
	void setOptionNodeFunction(unsigned int tmpNodeFunction);
	void setOptionThreadCombinatingMode(unsigned int iTCMode);
	void setPercentVerification(double tmpPercentVerifiy);
	void setTrainingRow(unsigned int iTmpRow);
	void setInputOffset(double tmpInputOffset);
	void sortNetwork();
	void shuffleTrainingData();
	void trainLine();
	void trainNetwork(bool bSilent=false);
	std::string getDataFileName();
	std::vector<std::string> getErrorList();
	std::vector<std::string> splitString(const std::string& strInput, const std::string& strDelimiter);

private:
	// Constants

	double E = 2.71828;

	// variables
	std::vector<double> vecValues = { 0.0 }; // vector containing values including input
	std::vector<double> vecExpectedValues = { 0.0 }; // vector containing training outputvalues
	std::vector<double> vecThresholds = { 0.0 };; // vector containing threshold values,Theshold also known as bias. used as automatic linear offset in calculation
	std::vector<std::vector<double>> vecWeights = { { 0.0 } }; // matrix containing weight between nodes
	std::vector<std::vector<double>> vvTrainingDataMatrix = { { 0.0 } }; // training data from file loaded into this matrix; // this should be moved to a new class. reducing memory size
	std::vector<double> vecCalcDelta = { 0.0 };

	std::string theAIDataFileName = "";
	bool bHasBeenConnected = false;
	bool bOptionShuffle = false;
	bool optionWeight = false;
	bool bOptionStatus = false;
	bool bOptionIO = false;
	bool bSilent = false;
	bool optionNoDeep = false;
	bool bOptionDisplayAllNodes = false;
	bool bOptionAutoGenerate = false;
	bool bOptionCSVGER = false;
	bool bOptionMaxIterationSet = false;
	bool initializationDone = false;
	bool bHistoricData = false;
	bool bFutureData = false;
	bool bBackpropagationActive = false;
	double dInputOffset = 0.0;	// set an input offset
	double dPercentVerification = 0.0;	// set the percentage of verification data
	unsigned int iThreadedCombinationMode = 0; // set the mode for combining data
	unsigned int iActivationFunction = 0;
	unsigned int iNumInputNodes = 2;	// basic number of input nodes
	unsigned int iNumRealInputNodes = 2;
	unsigned int iNumOutputNodes = 1;	// number of output nodes
	int iTimePreviousRows = 0;	// number of previous rows for time-dependent calculation
	unsigned int iTimeNumInputColumns = 0;	// number of columns for time-dependent calculation
	int iTimeNextRows = 0;
	unsigned int iTimeNumOutputColumns = 0;
	unsigned int iMaxIterations = 1000;
	unsigned int iCounter = 0;
	unsigned int iTrainingDataRow = 0;
	unsigned int iTrainingDataRowsMax = 0; //number of rows in training data
	double dlearningRate = 0.2; // the default learning rate
	double dWorstError = 0.0;
	unsigned int iWorstErrorRow = 0;
	std::string strInternalName = "UNKNOWN";
	std::vector<std::string> vTrainingDataColumns={ "0" };
	std::vector<unsigned int> inputDataPullList = { 0 };
	std::vector<unsigned int> vdNetworkTopology = { 0 };
	std::vector<unsigned int> viLayerActivationFunction = { 0 };
	std::vector<std::string> errorList = { "0" };
	std::vector<std::vector<double>> vvErrors={ {0.0} };
	
	// functions
	std::string generateFileOutput(std::string& strFileContents);
	std::string generateFileInput(std::string& strFileContents);
	bool recalculateInputDataPullList();
	bool throwFailure(std::string tmpError, bool doexit);
	double getTrainingDataValue(unsigned int row, unsigned int column);
	double updateWeightsInLayer(int tmpLayer);
	static inline void ReplaceAllStrings(std::string &str, const std::string& from, const std::string& to);
	std::string getExcelColumn(size_t stColumn);
	unsigned int getNumberOfInputNodes();
	unsigned int getNumberOfOutputNodes();
	unsigned int resizeVectors();
	int validLayer(int& tmpLayer, bool tmpRemoveOffset = false);

	std::string NodeFunctionXLS(unsigned int tmpNode, std::string tmpCalculatedInput);
	std::string NodeFunctionJS(unsigned int tmpNode, std::string tmpCalculatedInput);
};
