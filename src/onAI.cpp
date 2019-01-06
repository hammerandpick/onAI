// onAI.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <Windows.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <climits>
#include <algorithm>    // std::random_shuffle
#include <vector>
#include <time.h>
#include <mutex>
#include "AINetClass.h"

// variables
unsigned int iTimeNumInputColumns = 0;	// How many columns for previous calculation?
unsigned int iTimeNextElements = 0;
unsigned int iTimeNumOutputColumns = 0;
const int iThreads = 5;	// number of calculation threads
bool bTrainingDataRowsCounting = true;
double MOMENTUM = 0.0;
unsigned int theAIDataFilePos = 0;
char* theAIWeightsFileName = "weights.aiweights.csv";
char* cThisFileName = "";
const double VERSION = 0.20171118;
bool optionsAuto = false;
bool optionsWeightSave = false;
bool optionsAllNodes = false;
bool optionMaxIterationsSet = false;
bool optionsNoDeep = false; // turning off deep network

AINetClass aincNetwork; // dataContainer for the Network
std::mutex myMutex; // for multithreading

int main(int, char*[], char*[]);

// Functon prototypes
void trainingExample(AINetClass&, std::vector<double>&, std::vector<double>&);
bool modifyInputs(AINetClass&, std::string);
void pause();
void threadedCalculation(AINetClass ptrToNetwork, unsigned int iThreadID);
// pre-exit function
void leaveApplication();


/*******************************
/
/	begin main application
/
*******************************/

int main(int argc, char *argv[], char *env[]) {
	// Welcome Screen
	printf("Neural Network Program\n(%s)\n", argv[0]);
	printf("Version:%10.8f \n\n--- OPTIONS ---", VERSION);

	atexit(leaveApplication);

	// environment variable
	::cThisFileName = argv[0];
	
	// variables 
	int chooseMode = 0;
	int theFirstElement = 0;

	// checking parameters for programm
	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?") || !strcmp(argv[i], "/?") || !strcmp(argv[i], "-help") || !strcmp(argv[i], "/h") || !strcmp(argv[i], "/help"))
		{
			printf("Usage:\n%s [options] [filename]\n\n", argv[0]);
			printf("List of options:\n-a\tshow all nodes\n");
			printf("-auto\t non-interactive-mode\n");
			printf("-autonet\t auto-generate internal network\n");
			printf("-csvger\t csv german style (0,0;0,0)\n");
			printf("-file\t [-file filename]\tuse specified file\n");
			printf("-func-bip\t use bipolar sigmoid  activation function\n");
			printf("-func-tanh\t use tanh activation function\n");
			printf("-func-lin\t use linear activation on output layer function\n");
			printf("-func-mix\t use tanh activation within and sigmoid out function\n");
			printf("-func-relu\t use relu on inner and linear on outer node\n");
			printf("-io\tshow io data while calculating\n");
			printf("-learn\t [-learn number]\tset learning rate\n");
			printf("-nodeep\tturning off deep nodes\n");
			printf("-h\tshow (this) help screen and exit\n-s\tshow status\n-save\tsave the weights to a file\n");
			printf("-maxit\t [-iteration number]\tset maxiteration\n");
			printf("-shuffle\tshuffle data after each try\n");
			printf("-w\tshow weights when finished\n");
			exit(0);
		}
		else if (!strcmp(argv[i], "-auto")) { printf("using non-interactive-mode\n"); ::optionsAuto = true; }
		else if (!strcmp(argv[i], "-autonet")) { printf("auto-generate internal network\n"); ::optionsAuto = true; }
		else if (!strcmp(argv[i], "-s")) { printf("display Status enabled\n"); aincNetwork.setOptionStatus(true); }
		else if (!strcmp(argv[i], "-w")) { printf("display Weights enabled\n"); aincNetwork.setOptionWeight( true); }
		else if (!strcmp(argv[i], "-a")) { printf("display all nodes enabled\n"); ::optionsAllNodes = true; }
		else if (!strcmp(argv[i], "-csvger")) { printf("csv in german style\n"); aincNetwork.setOptionCSV(true); }
		else if (!strcmp(argv[i], "-io")) { printf("display I/O-Data enabled\n"); aincNetwork.setOptionIO(true); }
		else if (!strcmp(argv[i], "-nodeep")) { printf("deep network disabled.  ATTENTION output function may not work properly.\n"); aincNetwork.setOptionNoDeep(true); }
		else if (!strcmp(argv[i], "-save")) { printf("save Weights enabled\n"); ::optionsWeightSave = true; }
		else if (!strcmp(argv[i], "-func-tanh")) { printf("activation function set to tanh\n"); aincNetwork.setOptionNodeFunction(1); }
		else if (!strcmp(argv[i], "-func-mix")) { printf("activation function set to tanh for inner nodes\n"); aincNetwork.setOptionNodeFunction(5);}
		else if (!strcmp(argv[i], "-func-lin")) { printf("activation function set to linear for output nodes\n"); aincNetwork.setOptionNodeFunction(4); }
		else if (!strcmp(argv[i], "-func-bip")) { printf("activation function set to bipolardigmoid\n"); aincNetwork.setOptionNodeFunction(2); }
		else if (!strcmp(argv[i], "-func-relu")) { printf("activation function set to relu for inner nodes and linear for output nodes\n"); aincNetwork.setOptionNodeFunction(3); }
		else if (!strcmp(argv[i], "-file")) { printf("useing file %s\n", argv[min(i + 1, argc-1)]); aincNetwork.setDataFileName(argv[min(i + 1, argc - 1)]); }
		else if (!strcmp(argv[i], "-maxit")) { printf("setting maximum iterations to %i\n", atoi(argv[min(i + 1, argc - 1)])); optionMaxIterationsSet = true; aincNetwork.setMaxIterations(atoi(argv[min(i + 1, argc - 1)])); }
		else if (!strcmp(argv[i], "-learn")) { printf("setting learning rate to %f\n", atof(argv[min(i + 1, argc - 1)])); aincNetwork.setLearningRate(atof(argv[min(i + 1, argc - 1)])); }
		else if (!strcmp(argv[i], "-offset")) { printf("setting offset to %8.4f\n", atof(argv[min(i + 1, argc - 1)])); aincNetwork.setInputOffset(atof(argv[min(i + 1, argc - 1)])); }
		else if (!strcmp(argv[i], "-shuffle")) { printf("shuffling activated\n"); aincNetwork.setOptionShuffle(true); }
		else {
			//dismiss all other ones
		}
	}
	printf("--- OPTIONS END ---");

	if (!::optionsAuto)
	{
		// asking if standard or file mode
		printf("0 = Standard Mode (internal Data)\n");
		printf("1 = File Mode (Load %s)\n", aincNetwork.getDataFileName().c_str());
		printf("Please select mode:");
		while (!(std::cin >> chooseMode))
		{
			std::cout << "Only Numbers!" << std::endl;
			std::cin.clear();
			std::cin.ignore(std::cin.rdbuf()->in_avail());
		}
	}

	if (((chooseMode == 1) || (::optionsAuto)) && (0 != strcmp("", aincNetwork.getDataFileName().c_str())))
	{
		std::string theLine;

		printf("Read training data file.\n Read %i lines of which %i were erroneus.",aincNetwork.getTrainingDataRowsMax(), aincNetwork.LoadTrainingDataFile());

	}
	else
	{
		aincNetwork.setNumInputNodes(2);
		aincNetwork.setNumOutputNodes(1);
	}

	aincNetwork.initialize();
	aincNetwork.connectNodes();
	aincNetwork.displayStatus();
	aincNetwork.setOptionThreadCombinatingMode(0);
	pause();

	//aincNetwork.trainNetwork();
	// initialize the network
	unsigned int iTmpMaxIt = aincNetwork.getMaxIterations();

	// todo remove next lines of code that all threads are indpendent.
	//aincNetwork.setMaxIterations(max(100,0.1*iTmpMaxIt));
	//aincNetwork.trainNetwork();
	//aincNetwork.setMaxIterations(0.9*iTmpMaxIt);
	
	if (iThreads > 0)
	{
		// initializing thread variables
		std::vector<std::thread> vtThread;
		vtThread.clear();
		//This statement will launch multiple threads in loop
		for (int i = 0; i < iThreads; ++i) {
			//start thread
			vtThread.push_back(std::thread(threadedCalculation, aincNetwork, i));
		}
		for (auto& thread : vtThread) {
			thread.join();
		}
		// get data from network 
	}
	printf("\nnumber of threads: %i", iThreads);
	// displaying Status
	aincNetwork.displayStatus();
	std::string sEnteredData="";
	while(modifyInputs(aincNetwork, sEnteredData) && !optionsAuto)
	{
		std::cin >> sEnteredData;
	}
	return 0;
}

void trainingExample(AINetClass& ptrAINetClass, std::vector<double>& ptrVNode, std::vector<double>& ptrVecExpectedValues)
{
	ptrAINetClass.setTrainingDataRowsMax(4);
	switch (ptrAINetClass.Counter() % 4)
	{
	case 0:
		ptrVNode[1] = 1;
		ptrVNode[2] = 1;
		ptrVecExpectedValues[ptrAINetClass.NUMNODES()] = 0;
		break;
	case 1:
		ptrVNode[1] = 0;
		ptrVNode[2] = 1;
		ptrVecExpectedValues[ptrAINetClass.NUMNODES()] = 1;
		break;
	case 2:
		ptrVNode[1] = 1;
		ptrVNode[2] = 0;
		ptrVecExpectedValues[ptrAINetClass.NUMNODES()] = 1;
		break;
	case 3:
		ptrVNode[1] = 0;
		ptrVNode[2] = 0;
		ptrVecExpectedValues[ptrAINetClass.NUMNODES()] = 0;
		break;
	default:
		break;
	}
}

bool modifyInputs(AINetClass& ptrAINetClass, std::string sEnterData)
{
	bool bRunAgain = true;
	std::transform(sEnterData.begin(), sEnterData.end(), sEnterData.begin(), ::tolower);
	if (sEnterData == "q" || sEnterData == "quit" || sEnterData == "exit") { bRunAgain = false; }
	if (sEnterData == "status") { ptrAINetClass.setOptionStatus(!ptrAINetClass.getOptionStatus()); ptrAINetClass.displayStatus(); }	// now toggling status and displaying data
	if (sEnterData == "io") { ptrAINetClass.setOptionIO(true); ptrAINetClass.displayIO(0.0); } // if options set displaying all net
	if (sEnterData == "node") { ptrAINetClass.displayAllNodes(0.0); } // if options ist set displaying all nodes
	if(sEnterData=="savenet"){aincNetwork.saveResultingNetwork();}
	if (!strcmp(sEnterData.c_str(), "switch")) 
	{
		// now check if data has been specified.
		std::cin >> sEnterData;
		unsigned int iDataset = atoi(sEnterData.c_str());
		if ((iDataset > 0)&& (iDataset<=ptrAINetClass.getTrainingDataRowsMax()))
		{
			aincNetwork.setTrainingRow(iDataset);
		}
	}
	else
	{
		// todo current values
	}
	printf("\nEnter>");
	return bRunAgain;
}

void pause()
{
	// pause the application
	if (!::optionsAuto)
	{
		std::cout << "\nPress ENTER to continue..." << std::endl;
		std::cin.ignore(10, '\n');
		std::cin.get();
	}
}

void threadedCalculation(AINetClass ptrToNetwork, unsigned int iThreadID)
{
	// calculation in threaded mode
	ptrToNetwork.setInternalName("Network_" + std::to_string(iThreadID));
	ptrToNetwork.connectNodes(true,iThreadID);
	if (iThreadID > 0)
	{
		ptrToNetwork.setOptionSilent(true);
		ptrToNetwork.setLearningRate((1.0 / iThreadID));
	}
	// training
	ptrToNetwork.trainNetwork();
	std::cout << "\nCalculation Thread " << iThreadID << " finished.";
	aincNetwork.combineNetworks(ptrToNetwork, ::myMutex, iThreadID);
}

void leaveApplication()
{
	// pre-exit funtion
	std::vector<std::string> tmpErrorList;
	std::vector<std::string> tmpNetworkErrorList;
	bool noErrors = true;
	tmpNetworkErrorList = aincNetwork.getErrorList();
	tmpErrorList.clear();
	tmpErrorList.insert(tmpErrorList.end(), tmpNetworkErrorList.begin(), tmpNetworkErrorList.end());
	printf("\n\nnow leaving application...");
	if(tmpErrorList.size()>0)
	{
		for (unsigned int i = 0; tmpErrorList.size(); i++)
		{
			printf("\nError %8i\t%s", i, tmpErrorList.at(i).c_str());
		}
		pause();	// show to user
	}
	printf("\nbye.");
}
