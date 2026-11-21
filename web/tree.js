// Vertical distance between consecutive moves in the tree, in px.
var LEVEL_HEIGHT = 70;

function CollapsibleTree (config) {
    this.config = config;
    this.defaultConfig = {
        width          : null,
        fullwidth      : true,
        height         : null,
        data           : false,
        duration       : 400,
        marginTop      : 20,
        marginBottom   : 20,
        marginLeft     : 120,
        marginRight    : 120,
        appendSelector : 'body'
    };

    this.initialize = function () {
        if (!(this.config instanceof Object)) {
            this.config = {};
        }
        this.config = Object.assign({}, this.defaultConfig, this.config);
    };
    
    this.collapse = function (d) {
        if (d.children && d.name) {
            d._children = d.children;
            d._children.forEach(this.collapse);
            d.children = null;
        }
        else if (d.children) {
            d.children.forEach(this.collapse);
        }
    }.bind(this);
    
    this.zoom = function () {
        var scale = d3.event.scale,
            translation = d3.event.translate;
        d3.select(this.config.appendSelector + " .drawarea")
            .attr("transform", "translate(" + translation + ")" +
                  " scale(" + scale + ")");
    }.bind(this);
    
    this.createTree = function (data) {
        var container = document.querySelector(this.config.appendSelector);
        var cw = this.config.width || container.clientWidth || 800;
        var ch = this.config.height || container.clientHeight || 600;

        var w = cw - this.config.marginRight - this.config.marginLeft;
        var h = ch - this.config.marginTop - this.config.marginBottom;        
        
        this.tree = d3.layout.tree().size([w, h]);
        this.diagonal = d3.svg.diagonal().projection(function(d) { return [d.x, d.y]; });
        
        d3.select(this.config.appendSelector).selectAll("svg").remove();

        this.svg = d3.select(this.config.appendSelector).append("svg")
                                .attr("width", this.config.fullwidth ? "100%" : (w + this.config.marginRight + this.config.marginLeft))
                                .attr("height", "100%")
                                .append("g")
                                .attr("class","drawarea")
                                .append("g")
                                .attr("transform", "translate(" + this.config.marginLeft + "," + this.config.marginTop + ")");
        
        this.root = data;
        this.root.x0 = w / 2;
        this.root.y0 = 0;
        
        this.update(this.root);
        
        this.zoomListener = d3.behavior.zoom()
                .scaleExtent([0.1, 3])
                .on("zoom", this.zoom);

        d3.select(this.config.appendSelector + " svg")
          .call(this.zoomListener);
    };

    this.focusNode = function(id) {
        if (!this.svg) return;
        // Use the positions the nodes were actually drawn at. Re-running the layout here
        // (as before) squeezed all depths into the panel height instead of LEVEL_HEIGHT px
        // per move, so the camera fell further behind the further into the game you went.
        var targetNode = null;
        this.svg.selectAll("g.node").each(function(d) { if (d.id === id) targetNode = d; });
        if (!targetNode) return;

        var container = document.querySelector(this.config.appendSelector);
        var cw = this.config.width || container.clientWidth || 800;
        var ch = this.config.height || container.clientHeight || 600;

        var scale = this.zoomListener.scale();
        var x = cw / 2 - (targetNode.x + this.config.marginLeft) * scale;
        var y = ch / 3 - (targetNode.y + this.config.marginTop) * scale;

        d3.select(this.config.appendSelector + " .drawarea").interrupt().transition().duration(200)
            .attr("transform", "translate(" + x + "," + y + ") scale(" + scale + ")");
        this.zoomListener.translate([x, y]);
    }.bind(this);
    
    this.onClick = function (d) {
        if (d.name) {
            if (window.goToMoveById && d.id && d.id !== 'root') {
                window.goToMoveById(d.id);
            }
        }
    }.bind(this);
    
    this.update = function (source) {
          var nodes    = this.tree.nodes(this.root).reverse(),
              links    = this.tree.links(nodes),
              i        = 0,
              duration = this.config.duration;

          nodes.forEach(function(d) { d.y = d.depth * LEVEL_HEIGHT; }.bind(this));

          var node = this.svg.selectAll("g.node")
                             .data(nodes, function(d) { return d.id || (d.id = ++i); });

          var nodeEnter = node.enter().append("g")
                                      .attr("id", function(d) { return "tree-node-" + d.id; })
                                      .attr("class",  function(d){ return "node " + "node_" + (d.type || "MAINLINE"); })
                                      .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
                                      .on("click", this.onClick);
        
          nodeEnter.append("circle")
                   .attr("r", 1e-6)
                   .attr("class", function(d) {
                        return d._children ? "children" : "";
                    });

          nodeEnter.append("text")
                   .attr("x", 12)
                   .attr("dy", ".35em")
                   .attr("text-anchor", "start")
                   .text(function(d) { return d.name; })
                   .style("fill-opacity", 1e-6);

          var nodeUpdate = node.transition()
                               .duration(duration)
                               .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

          nodeUpdate.select("circle")
                    .attr("r", 6)
                    .attr("class", function(d) {
                        return d._children ? "children" : "";
                    });

          nodeUpdate.select("text")
              .style("fill-opacity", 1);

          var nodeExit = node.exit().transition()
                             .duration(duration)
                             .attr("transform", function(d) { return "translate(" + source.x + "," + source.y + ")"; })
                             .remove();

          nodeExit.select("circle")
                  .attr("r", 1e-6);

          nodeExit.select("text")
                  .style("fill-opacity", 1e-6);

          var link = this.svg.selectAll("path.link")
                             .data(links, function(d) { return d.target.id; });

          link.enter().insert("path", "g")
                      .attr("class",  function(d){ return "link" + " link_" + (d.target.linkType || "MAINLINE"); })
                      .attr("d", function(d) {
                          var o = {x: source.x0, y: source.y0};
                          return this.diagonal({source: o, target: o});
                      }.bind(this));

          link.transition()
              .duration(duration)
              .attr("d", this.diagonal);

          link.exit().transition()
              .duration(duration)
              .attr("d", function(d) {
                  var o = {x: source.x, y: source.y};
                  return this.diagonal({source: o, target: o});
              }.bind(this))
              .remove();

          nodes.forEach(function(d) {
              d.x0 = d.x;
              d.y0 = d.y;
          });
    };
    
    this.render = function () {
        if (this.config.data instanceof Object) {
            this.createTree(this.config.data);
        }        
    };
        
    this.initialize();
}
