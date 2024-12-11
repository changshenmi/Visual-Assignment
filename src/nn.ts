/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * 神经网络中的一个节点。每个节点都有一个状态
 * (总输入、输出及其各自的导数)，这个状态在每次
 * 前向和反向传播后都会更新。
 */
export class Node {
  id: string;
  inputLinks: Link[] = []; // 输入连接列表
  bias = 0.1; // 偏置
  outputs: Link[] = []; // 输出连接列表
  totalInput: number; // 总输入
  output: number; // 输出
  outputDer = 0; // 相对于输出的误差导数
  inputDer = 0; // 相对于总���入的误差导数
  accInputDer = 0; // 自上次更新以来累积的相对于总输入的误差导数
  numAccumulatedDers = 0; // 自上次更新以来累积的相对于总输入的误差导数数量
  activation: ActivationFunction; // 激活函数

  /**
   * 使用提供的id和激活函数创建一个新节点。
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /* 计算节点的输出 */
  updateOutput(): number {
    // 存储节点的总输入
    this.totalInput = this.bias;//总输入初始化为偏置
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;//输出 = activation(Σ(weight * input) + bias)
    }
    this.output = this.activation.output(this.totalInput);//激活函数
    return this.output;
  }
}

/**
 * 误差函数及其导数
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** 节点的激活函数及其导数 */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** 计算网络中给定权重的惩罚成本的函数 */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** 内置误差函数 */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),//平方误差，
    der: (output: number, target: number) => output - target//平方误差导数
  };
}

/** TANH的polyfill实现 */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** 内置激活函数 */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static LEAKY_RELU: ActivationFunction = {
    output: x => x > 0 ? x : 0.2 * x,  // alpha = 0.2
    der: x => x > 0 ? 1 : 0.2  // 导数：x>0时为1，x<=0时为alpha
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
}

/** 内置正则化函数L1 和 L2 */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * 神经网络中的一个连接。每个连接都有一个权重以及源节点和
 * 目标节点。同时它还有一个内部状态（相对于特定输入的误差导数），
 * 这个状态在反向传播运行后会更新。
 */
export class Link {
  id: string;
  source: Node;//源节点
  dest: Node;//目标节点
  weight = Math.random() - 0.5;//权重
  isDead = false;//是否死亡
  /** 相对于此权重的误差导数 */
  errorDer = 0;
  /** 自上次更新以来累积的误差导数 */
  accErrorDer = 0;
  /** 自上次更新以来累积的导数数量 */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;//使用内置的正则化函数

  /**
   * 构造神经网络中的一个连接，使用随机权重初始化。
   *
   * @param source 源节点
   * @param dest 目标节点
   * @param regularization 计算此权重惩罚的正则化函数。
   *     如果为null，则不会进行正则化。
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * 构建神经网络。
 *
 * @param networkShape 网络的形状。例如 [1, 2, 3, 1] 表示
 *   网络将有1个输入节点，第一隐藏层2个节点，
 *   第二隐藏层3个节点，1个输出节点。
 * @param activation 每个隐藏节点的激活函数
 * @param outputActivation 输出节点的激活函数
 * @param regularization 计算网络中给定权重（参数）惩罚的正则化函数。
 *     如果为null，则不会进行正则化。
 * @param inputIds 输入节点的id列表
 */
export function buildNetwork(
    networkShape: number[], 
    activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
  let numLayers = networkShape.length;//层数
  let id = 1;//节点id从1开始
  /** 层的列表，每层都是节点的列表 */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;//是输出层是true
    let isInputLayer = layerIdx === 0;//是输入层是true，通过layerId

    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // 添加从前一层节点到此节点的连接
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * 通过提供的网络对提供的输入进行前向传播。
 * 此方法修改网络的内部状态 - 网络中每个���点的
 * 总输入和输出。
 *
 * @param network 神经网络
 * @param inputs 输入数组。其长度应该与网络中的输入节点数量匹配
 * @return 网络的最终输出
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("输入数量必须与输入层中的节点数量匹配");
  }
  // 更新输入层
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // 更新该层中的所有节点
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * 使用提供的目标和前一次前向传播计算的输出
 * 运行反向传播。此方法修改网络的内部状态 - 
 * 相对于每个节点的误差导数，以及网络中每个权重。
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // 输出节点是一个特殊情况。我们使用用户定义的误差函数来计算导数。
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // 从后向前遍历层
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // 计算每个节点相对于以下方面的误差导数：
    // 1) 其总输入
    // 2) 其每个输入权重
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // 计算相对于进入节点的每个权重的误差导数
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // 计算相对于每个节点输出的误差导数
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/**
 * 使用先前累积的误差导数更新网络的权重
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // 更新节点的偏置
      if (node.numAccumulatedDers > 0) {
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // 更新进入此节点的权重
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization ?
            link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          // 基于 dE/dw 更新权重
          link.weight = link.weight -
              (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          // 基于正则化进一步更新权重
          let newLinkWeight = link.weight -
              (learningRate * regularizationRate) * regulDer;
          if (link.regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // 由于正则化项，权重穿过了0。将其设置为0
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** 遍历网络中的每个节点 */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** 返回网络中的输出节点 */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
