import * as d3 from 'd3';

interface NodeValue {
  id: string;
  forward: number;
  backward: number;
}

interface LayerValues {
  layerId: number;
  nodes: NodeValue[];
}

export class BPVisualization {
  private container: d3.Selection<any>;
  private data: LayerValues[] = [];
  private table: d3.Selection<any>;
  private width = 600;
  private showForward: boolean = true;
  private title: d3.Selection<any>;
  
  constructor(containerId: string) {
    this.container = d3.select(`#${containerId}`)
      .append('div')
      .attr('class', 'bp-vis-container');

    this.title = d3.select('.bp-vis-section h4 span');

    this.container.append('button')
      .attr('class', 'bp-toggle-btn')
      .text('切换正向/反向传播')
      .on('click', () => {
        this.showForward = !this.showForward;
        this.title.text(this.showForward ? "正向传播过程" : "反向传播过程");
        this.render();
      });
      
    this.table = this.container.append('table')
      .attr('class', 'bp-vis-table');
  }

  public updateData(networkValues: LayerValues[]) {
    this.data = networkValues;
    this.render();
  }

  private render() {
    this.table.selectAll('*').remove();
    
    // 过滤掉空的层
    const nonEmptyData = this.data.filter(layer => layer.nodes.length > 0);
    
    const totalColumns = nonEmptyData.length * 2 - 1;
    this.table.attr('class', `bp-vis-table columns-${totalColumns}`);
    
    // 添加说明文字
    const description = this.container.selectAll('.bp-description').data([0]);
    description.enter()
      .append('div')
      .attr('class', 'bp-description');
    
    description.html(this.showForward ? 
      "正向传播值：表示每个神经元的输出值（经过激活函数后的值）" : 
      "反向传播值：表示损失函数对每个神经元输入的梯度值（用于更新权重）");
    
    const thead = this.table.append('thead').append('tr');
    
    // 添加层标题和空白列
    nonEmptyData.forEach((layer, i) => {
      // 添加层标题
      thead.append('th')
        .text(this.getLayerName(layer.layerId))
        .attr('class', 'layer-header');
        
      // 在非最后一层后添加空白列
      if (i < nonEmptyData.length - 1) {
        thead.append('th')
          .attr('class', 'spacer-header')
          .text('');
      }
    });

    const maxNodes = Math.max(...nonEmptyData.map(layer => layer.nodes.length));
    const tbody = this.table.append('tbody');
    
    for (let nodeIdx = 0; nodeIdx < maxNodes; nodeIdx++) {
      const row = tbody.append('tr');
      
      nonEmptyData.forEach((layer, i) => {
        // 添加节点单元格
        const cell = row.append('td')
          .attr('class', 'node-cell');
        
        if (nodeIdx < layer.nodes.length) {
          const node = layer.nodes[nodeIdx];
          const isInputLayer = layer.layerId === 0;
          
          cell.append('div')
            .attr('class', 'node-name')
            .text(this.getNodeName(node.id, isInputLayer));
          
          // 只为非输入层节点显示值
          if (!isInputLayer) {
            const value = this.showForward ? node.forward : node.backward;
            cell.append('div')
              .attr('class', 'node-value')
              .text(this.formatValue(value))
              .style('color', this.getValueColor(value));
          }
        }
        
        // 在非最后一层后添加空白单元格
        if (i < nonEmptyData.length - 1) {
          row.append('td')
            .attr('class', 'spacer-cell')
            .text('');
        }
      });
    }
  }

  private getValueColor(value: number): string {
    if (Math.abs(value) < 0.0001) return '#999';
    return value > 0 ? '#0877bd' : '#f59322';
  }

  private getLayerName(layerId: number): string {
    if (layerId === 0) return "输入层";
    if (layerId === this.data.length - 1) return "输出层";
    return `隐藏层 ${layerId}`;
  }

  private getNodeName(nodeId: string, isInput: boolean): string {
    if (isInput) {
      // 输入层节点名称处理
      const featureMap: {[key: string]: string} = {
        'x': 'X₁',
        'y': 'X₂',
        'xSquared': 'X₁²',
        'ySquared': 'X₂²',
        'xTimesY': 'X₁X₂',
        'sinX': 'sin(X₁)',
        'sinY': 'sin(X₂)'
      };
      return featureMap[nodeId] || nodeId;
    }
    return nodeId.indexOf('x') >= 0 ? nodeId : `节点 ${nodeId}`;
  }

  private formatValue(value: number): string {
    if (Math.abs(value) < 0.0001) return "0.0000";
    return value.toFixed(4);
  }

  public clear() {
    this.data = [];
    this.render();
  }
} 