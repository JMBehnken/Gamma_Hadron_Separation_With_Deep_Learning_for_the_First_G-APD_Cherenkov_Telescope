/*
  Code to generate cube grid to hexagon grid diagram on http://www.redblobgames.com/articles/grids/
  Copyright 2012 Red Blob Games <redblobgames@gmail.com>
  License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>
*/

function makeCubeGridDiagram(id) {
    var SQRT_3_2 = Math.sqrt(3)/2;
    var base_z = -3;
    var limit = 4;
    var svg = d3.select("#" + id);
    var root = svg.append('svg:g')
        .attr('transform', "translate(167.5, 184.5)");

    var cube_coords = [];
    for (var q = -2; q < 2; q++) {
        for (var r = -2; r < 2; r++) {
            for (var s = -2; s < 2; s++) {
                var z = q + r + s;
                if (base_z <= z && z <= limit) {
                    cube_coords.push({q: q, r: r, s: s, z: z});
                }
            }
        }
    }

    var scale = 40;
    function toScreen(q, r, s) {
        return new ScreenCoordinate(scale * (r-q) * SQRT_3_2, scale * (0.5*(r+q) - s));
    }

    // Axes
    root.append('line')
        .attr('class', "z-axis")
        .attr('x1', toScreen(0, 0, -5).x).attr('y1', toScreen(0, 0, -5).y)
        .attr('x2', toScreen(0, 0, 5).x).attr('y2', toScreen(0, 0, 5).y);
    root.append('text')
        .text("-z")
        .attr('transform', "translate(" + toScreen(0, 0, 4.2) + ") translate(10, 0)");
    root.append('text')
        .text("+z")
        .attr('transform', "translate(" + toScreen(0, 0, -4.5) + ") translate(-10, 0)");
    root.append('line')
        .attr('class', "y-axis")
        .attr('x1', toScreen(0, -5, 0).x).attr('y1', toScreen(0, -5, 0).y)
        .attr('x2', toScreen(0, 5, 0).x).attr('y2', toScreen(0, 5, 0).y);
    root.append('text')
        .text("-y")
        .attr('transform', "translate(" + toScreen(0, 4.5, 0) + ") translate(-7, 7)");
    root.append('text')
        .text("+y")
        .attr('transform', "translate(" + toScreen(0, -4.5, 0) + ") translate(7, -7)");
    root.append('line')
        .attr('class', "x-axis")
        .attr('x1', toScreen(-5, 0, 0).x).attr('y1', toScreen(-5, 0, 0).y)
        .attr('x2', toScreen(5, 0, 0).x).attr('y2', toScreen(5, 0, 0).y);
    root.append('text')
        .text("-x")
        .attr('transform', "translate(" + toScreen(4.5, 0, 0) + ") translate(-7, -7)");
    root.append('text')
        .text("+x")
        .attr('transform', "translate(" + toScreen(-4.5, 0, 0) + ") translate(7, 7)");

    // Draw the cubes in svg
    var cubes = root.selectAll('g').data(cube_coords);
    var offsets = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1]];
    cubes.enter()
         .append('g')
         .attr('class', "cube onscreen");
    cubes.append('polygon').attr('class', 'full');
    cubes.append('polygon').attr('class', 'shade');
    cubes.append('path');

    function coord(q, r, s) {
        var p = toScreen(q, r, s);
        return p.x + "," + p.y;
    }

    cubes.select('path')
         .attr('d', function (cube) {
        var p1 = toScreen(cube.q, cube.r, cube.s);
        var p2 = toScreen(cube.q + 1, cube.r + 1, cube.s);
        var p3 = toScreen(cube.q + 1, cube.r, cube.s + 1);
        var p4 = toScreen(cube.q, cube.r + 1, cube.s + 1);
        return ["M", p1, "L", p2,
                "M", p1, "L", p3,
                "M", p1, "L", p4].join(" ");
    });
    
    cubes.select('polygon.full')
         .attr('points', function(cube) {
        var points = [];
        offsets.forEach(function (offset) {
            points.push(coord(cube.q + offset[0], cube.r + offset[1], cube.s + offset[2]));
        });
        return points.join(" ");
    });
    
    cubes.select('polygon.shade')
         .attr('points', function(cube) {
        var points = [coord(cube.q, cube.r, cube.s)];
        offsets.slice(1, 4).forEach(function (offset) {
            points.push(coord(cube.q + offset[0], cube.r + offset[1], cube.s + offset[2]));
        });
        return points.join(" ");
    });
    
    // Transition effect: some cubes will be offscreen, depending on
    // the value of 'limit'.
    function redraw() {
        // I put some extra factors in the offscreen test so that the cubes separate out a little bit
        cubes
            .classed('base', function(d) { return d.z == base_z && limit == base_z; })
            .classed('offscreen', function(d) { return d.z > base_z && (d.z + d.r * 0.04 - d.q * 0.03 + d.r * 0.05) >= limit; });
    }

    // Animation is implemented by creating an attribute on the svg,
    // and then using d3 transitions to adjust that attribute. Then I
    // have a separate callback using d3.timer that reads the
    // attribute and sets the CSS classes appropriately. Then CSS
    // transition effects perform the actual animation. Is there a
    // cleaner way to do this? I hope so but I haven't found it yet.
    svg.attr('data-limit', limit);
    var animation_complete = true;
    function redraw_if_animating() {
        limit = svg.attr('data-limit');
        redraw();
        if (animation_complete) { setTimeout(redraw, 1000); }
        return animation_complete;
    }

    redraw.to_cube = function() {
        animation_complete = false;
        svg.transition().ease('linear').duration(6000).attr('data-limit', 4)
            .each('end', function() { animation_complete = true; });
        d3.timer(redraw_if_animating);
        d3.selectAll("button.cube-to-hex-cubes").classed('highlight', true);
        d3.selectAll("button.cube-to-hex-hexagons").classed('highlight', false);
    }

    redraw.to_hex = function() {
        animation_complete = false;
        svg.transition().ease('linear').duration(6000).attr('data-limit', base_z)
            .each('end', function() { animation_complete = true; });
        d3.timer(redraw_if_animating);
        d3.selectAll("button.cube-to-hex-cubes").classed('highlight', false);
        d3.selectAll("button.cube-to-hex-hexagons").classed('highlight', true);
    }

    d3.selectAll("button.cube-to-hex-cubes").classed('highlight', true);
    redraw();
    return redraw;
}


var diagram_cube_to_hex = makeCubeGridDiagram('cube-to-hex');
